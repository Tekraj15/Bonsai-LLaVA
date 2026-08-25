import argparse
import os
import sys
import yaml
import torch
from transformers import (
    TrainingArguments, AutoTokenizer, AutoProcessor, Trainer,
    TrainerCallback, set_seed,
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, concatenate_datasets

# Tokenizers fork-safety under dataloader workers.
# NOTE: PYTORCH_MPS_HIGH_WATERMARK_RATIO is deliberately NOT set here. The value
# 0.0 means "no allocation ceiling", which lets MPS push an 8GB machine into swap
# and silently turns a 2s step into a 30s one. Leave PyTorch's default in place.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent
from src.datasets.llava_dataset import LLaVADataset
from src.datasets.data_collator import CustomeDataCollatorForLlava


class BonsaiSFTTrainer(Trainer):
    """
    Trainer with two behaviour changes for this model.

    1. Length-grouped batching is fed our cheap, text-only lengths. Without this,
       HF derives lengths by iterating the dataset — which for a raw-image
       dataset decodes and preprocesses every image before step 1.
    2. Checkpoints contain only trainable weights. `BonsaiStudent` is a plain
       nn.Module, so Trainer's default `_save` serialises the entire ~1.1GB state
       dict (frozen Qwen + SigLIP) into every checkpoint. That is what previously
       exhausted local disk, so `_save` is overridden rather than merely
       supplemented by a callback.
    """
    def __init__(self, *args, bonsai_tokenizer=None, bonsai_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonsai_tokenizer = bonsai_tokenizer
        self.bonsai_processor = bonsai_processor

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        save_components(self.model, output_dir,
                        self.bonsai_tokenizer, self.bonsai_processor)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print(f"[checkpoint] trainable weights only -> {output_dir}")

    def _get_train_sampler(self, *args, **kwargs):
        if self.args.group_by_length and hasattr(self.train_dataset, "lengths"):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self.train_dataset.lengths,
            )
        return super()._get_train_sampler(*args, **kwargs)


def save_components(student_model, output_dir, tokenizer=None, processor=None):
    """Persist the LoRA adapter, projector, tokenizer and processor."""
    os.makedirs(output_dir, exist_ok=True)
    student_model.language_model.save_pretrained(output_dir)
    torch.save(student_model.projector.state_dict(),
               os.path.join(output_dir, "projector.pt"))
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)


def build_student(config, device_dtype=torch.float16):
    """
    Build the student with a numerically safe precision split:
      - frozen backbone (Qwen + SigLIP) in FP16  -> memory
      - trainable params (LoRA + projector) in FP32 -> stable Adam updates

    Casting the trainable params to FP16 (as the previous revision did) puts
    gradients near the FP16 underflow floor (~6e-5) and pushes Adam's eps=1e-8
    below representable resolution, so updates silently degrade or NaN.
    """
    student_model = BonsaiStudent(
        vision_model_name=config["vision_tower"],
        language_model_name=config["model_name_or_path"],
        load_in_4bit=config.get("load_in_4bit", False),
    )

    # Frozen backbone -> FP16.
    student_model.language_model.to(device_dtype)
    student_model.vision_tower.to(device_dtype)
    for param in student_model.language_model.parameters():
        param.requires_grad = False

    # LoRA adapters.
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model.language_model = get_peft_model(student_model.language_model, peft_config)

    # PEFT creates adapters in the base layer's dtype (FP16 here); promote the
    # trainable ones back to FP32.
    for name, param in student_model.language_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Projector trains from scratch — keep it FP32 throughout.
    student_model.projector.to(torch.float32)
    for param in student_model.projector.parameters():
        param.requires_grad = True

    return student_model


def load_dataset_for_training(config, tokenizer, image_processor, max_samples=None):
    """
    Prefer raw images (the vision tower is <2% of step FLOPs, and dataloader
    workers hide the decode cost). Fall back to a pre-computed feature cache only
    if one exists AND it was built after the labeling fixes.
    """
    cache_root = config.get("feature_cache_dir")
    use_cache = bool(cache_root) and os.path.exists(cache_root)

    if use_cache:
        print(f"Loading Sharded Datasets from {cache_root}...")
        shard_dirs = sorted(
            os.path.join(cache_root, d)
            for d in os.listdir(cache_root)
            if d.startswith("shard_") and os.path.isdir(os.path.join(cache_root, d))
        )
        if shard_dirs:
            loaded = []
            for s_path in shard_dirs:
                try:
                    loaded.append(load_from_disk(s_path))
                except Exception as e:
                    print(f"Warning: Could not load shard {s_path}: {e}")
            if loaded:
                dataset = concatenate_datasets(loaded)
                dataset.set_format("torch")
                print(f"Successfully loaded {len(dataset)} cached samples.")
                return dataset
        print("Feature cache present but unusable; falling back to raw images.")

    print("Loading raw-image dataset...")
    return LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.get("max_length", 640),
        max_samples=max_samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Bonsai-LLaVA SFT training")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N optimizer steps (overrides epochs). "
                             "Use a small value to smoke-test the pipeline.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Train on only the first N usable samples.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps. Lower it for a short "
                             "watchable run; the config default of 32 means one "
                             "optimizer step costs 32 forward/backward passes.")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def train(args=None):
    args = args or parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides win over the config file.
    for key, value in [
        ("max_steps", args.max_steps),
        ("max_train_samples", args.max_train_samples),
        ("output_dir", args.output_dir),
        ("per_device_train_batch_size", args.batch_size),
        ("gradient_accumulation_steps", args.grad_accum),
        ("dataloader_num_workers", args.num_workers),
    ]:
        if value is not None:
            config[key] = value

    seed = config.get("seed", 42)
    set_seed(seed)

    print("--- Training Environment Diagnostics ---")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Selected Device: {device} | Seed: {seed}")
    print("----------------------------------------")

    print("Loading Student Model...")
    student_model = build_student(config)
    student_model.print_trainable_parameters()

    # Tokenizer: keep Qwen's native pad token (<|endoftext|>). Aliasing pad to eos
    # (<|im_end|>) makes padding indistinguishable from end-of-turn and poisons
    # any eos-based logic downstream.
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no native pad token; set one explicitly.")
    print(f"pad_token={tokenizer.pad_token!r} eos_token={tokenizer.eos_token!r}")

    image_processor = AutoProcessor.from_pretrained(config["vision_tower"], use_fast=True)

    # Gradient checkpointing needs the frozen backbone's inputs to require grad,
    # otherwise the recomputed segments have nothing to backprop through and the
    # LoRA/projector gradients come out empty.
    if config.get("gradient_checkpointing", False):
        student_model.language_model.enable_input_require_grads()
        print("Gradient checkpointing enabled (input grads on).")

    # Length grouping buys nothing at batch 1 — a batch of one has nothing to pad
    # against — while ordering longest-first, which drives peak memory into swap
    # on the very first steps. Measured: 480-token samples cost 107s/step there.
    if config["per_device_train_batch_size"] == 1 and config.get("group_by_length"):
        print("batch_size=1: disabling group_by_length (no benefit, worst-case-first ordering).")
        config["group_by_length"] = False

    dataset = load_dataset_for_training(
        config, tokenizer, image_processor,
        max_samples=config.get("max_train_samples"),
    )
    print(f"Training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
        num_train_epochs=config["num_train_epochs"],
        max_steps=config.get("max_steps", -1),
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config.get("max_grad_norm", 1.0),
        # Model weights are already FP16/FP32 by design; autocast + GradScaler on
        # top of that is both redundant and unsupported on this torch/MPS build.
        bf16=False,
        fp16=False,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        group_by_length=config.get("group_by_length", True),
        length_column_name="length",
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        save_safetensors=False,
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        seed=seed,
        report_to=config["report_to"],
        remove_unused_columns=False,
    )

    trainer = BonsaiSFTTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CustomeDataCollatorForLlava(tokenizer=tokenizer),
        bonsai_tokenizer=tokenizer,
        bonsai_processor=image_processor,
    )

    print("Starting SFT Training...")
    trainer.train()

    print("Saving Model Components...")
    save_components(student_model, config["output_dir"], tokenizer, image_processor)
    print(f"Model components saved to {config['output_dir']}")


if __name__ == "__main__":
    train()
