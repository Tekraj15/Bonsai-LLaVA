"""
Pipeline validation gate: overfit a handful of samples to near-zero loss.

This is the cheapest test that distinguishes "the pipeline is wired correctly"
from "the pipeline trains a broken objective". A model that cannot memorise 32
samples has a bug in labels, masking, dtypes or the optimizer — no amount of GPU
time will fix that, so run this before every real training run.

    python scripts/sanity_overfit.py --samples 32 --steps 60

PASS criteria:
  1. Final loss < 0.5 (memorisation achieved).
  2. Loss decreased monotonically-ish from the initial value.
  3. Generated text on a training image is coherent and on-topic
     (not an endless stream of <|im_end|>).
"""
import argparse
import os
import sys
import time

import torch
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoProcessor, AutoTokenizer, set_seed

from src.datasets.data_collator import CustomeDataCollatorForLlava
from src.datasets.llava_dataset import LLaVADataset
from src.utils.constants import IGNORE_INDEX, SYSTEM_PROMPT
from src.utils.mps_compat import generation_device
from scripts.train_sft import build_student


def report_batch_stats(batch, tokenizer):
    labels = batch["labels"]
    total = labels.numel()
    supervised = int((labels != IGNORE_INDEX).sum())
    pad_positions = int((batch["attention_mask"] == 0).sum())
    supervised_pad = int(((labels != IGNORE_INDEX) & (batch["attention_mask"] == 0)).sum())

    print("\n--- Batch label contract ---")
    print(f"  shape                : {tuple(labels.shape)}")
    print(f"  supervised positions : {supervised}/{total} ({supervised / total:.1%})")
    print(f"  padded positions     : {pad_positions}")
    print(f"  supervised AND padded: {supervised_pad}  (must be 0)")
    assert supervised_pad == 0, "Loss is being computed on padding tokens."
    assert supervised > 0, "No supervised tokens in batch."

    first = labels[0]
    text = tokenizer.decode([t for t in first if t != IGNORE_INDEX])
    print(f"  supervised text[0]   : {text[:160]!r}")
    print("----------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    args = parser.parse_args()

    set_seed(42)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | samples={args.samples} steps={args.steps}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    image_processor = AutoProcessor.from_pretrained(config["vision_tower"], use_fast=True)
    print(f"pad={tokenizer.pad_token!r} eos={tokenizer.eos_token!r} "
          f"distinct={tokenizer.pad_token_id != tokenizer.eos_token_id}")

    dataset = LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.get("max_length", 640),
        max_samples=args.samples,
    )
    if len(dataset) == 0:
        print("No usable samples — run scripts/fetch_subset_images.py first.")
        sys.exit(1)
    print(f"Loaded {len(dataset)} samples.")

    collator = CustomeDataCollatorForLlava(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )

    model = build_student(config)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable), \
        "Trainable params must be FP32 for stable Adam updates."
    print(f"All {len(trainable)} trainable tensors are FP32. ✓")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    # Cosine decay to zero. Without it a constant LR high enough to memorise
    # quickly also destabilises once the loss is low: an earlier run reached
    # 0.12 at step 50 and then climbed back to 1.26 by step 79 with grad norms
    # rising 2.6 -> 20.1. Decay keeps the fast early progress and lands at the
    # minimum instead of oscillating past it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    model.train()

    losses = []
    step = 0
    start = time.time()
    printed_stats = False
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            if not printed_stats:
                report_batch_stats({k: v.cpu() for k, v in batch.items()}, tokenizer)
                printed_stats = True

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            if step % 10 == 0 or step == args.steps - 1:
                print(f"  step {step:3d} | loss {loss.item():7.4f} | "
                      f"grad_norm {float(grad_norm):7.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
            if not torch.isfinite(loss):
                print("\nFAIL: loss became non-finite (NaN/Inf) — precision bug.")
                sys.exit(1)
            step += 1

    elapsed = time.time() - start
    initial = sum(losses[:3]) / min(3, len(losses))
    final = sum(losses[-3:]) / min(3, len(losses))
    sec_per_step = elapsed / max(step, 1)

    best = min(losses)
    print(f"\n--- Results ---")
    print(f"  initial loss : {initial:.4f}")
    print(f"  best loss    : {best:.4f}  (step {losses.index(best)})")
    print(f"  final loss   : {final:.4f}")
    print(f"  wall clock   : {elapsed:.1f}s ({sec_per_step:.2f}s/step, "
          f"batch={args.batch_size})")

    # Generation spot-check on a training sample. Runs on the generation device
    # (CPU on macOS 13, where MPS aborts inside generate()); training above
    # stays on MPS.
    model.eval()
    gen_device = generation_device()
    if gen_device != device:
        print(f"\nMoving model to {gen_device} for the generation spot-check...")
        model.to(gen_device)
        model.language_model.to(gen_device)
        model.vision_tower.to(gen_device)
        model.projector.to(gen_device)
    device = gen_device
    sample = dataset[0]
    convs = dataset.data[0]["conversations"]
    question = convs[0]["value"].replace("<image>", "").strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            pixel_values=sample["pixel_values"].unsqueeze(0).to(device),
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\n--- Generation spot-check ---")
    print(f"  Q          : {question[:120]}")
    print(f"  expected   : {convs[1]['value'][:120]}")
    print(f"  generated  : {answer[:200]!r}")

    passed = final < 0.5 and final < initial
    print(f"\n{'PASS' if passed else 'FAIL'}: final loss {final:.4f} "
          f"(target < 0.5, initial was {initial:.4f})")
    if not passed:
        print("The pipeline cannot memorise a tiny batch — investigate labels, "
              "masking, dtypes or LR before spending real training time.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
