import os
import sys
import yaml
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoProcessor, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# environment variables for MPS memory and Tokenizer parallelism
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent
from src.datasets.llava_dataset import LLaVADataset
from src.datasets.data_collator import DataCollatorForDistillation

def train():
    # Load Config
    with open("configs/qlora_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Device Check
    print(f"--- Training Environment Diagnostics ---")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Selected Device: {device}")
    print(f"----------------------------------------")

    print("Loading Student Model...")
    # 1. Load Student
    # We use load_in_4bit=True to keep memory usage low, as defined in the class
    student_model = BonsaiStudent(
        vision_model_name=config["vision_tower"],
        language_model_name=config["model_name_or_path"],
        load_in_4bit=True 
    )

    # Enable Gradient Checkpointing
    student_model.language_model.gradient_checkpointing_enable()
    student_model.language_model = prepare_model_for_kbit_training(student_model.language_model)

    # 2. Apply LoRA to Student Language Model
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model.language_model = get_peft_model(student_model.language_model, peft_config)
    student_model.print_trainable_parameters()

    # Ensure Projector is trainable
    for param in student_model.projector.parameters():
        param.requires_grad = True

    # 3. Load Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token 
    
    image_processor = AutoProcessor.from_pretrained(config["vision_tower"], use_fast=True)

    # 4. Load Dataset
    dataset = LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # 5. Training Arguments
    # Force FP16 for MPS compatibility
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        bf16=False, 
        fp16=False, # Disable mixed precision to bypass accelerate check (model is already FP16)
        tf32=False,
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        dataloader_num_workers=config["dataloader_num_workers"],
        report_to=config["report_to"],
        remove_unused_columns=False 
    )

    # 6. Trainer (Standard HF Trainer)
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForDistillation(tokenizer=tokenizer),
    )

    print("Starting SFT Training...")
    trainer.train()

    print("Saving Model...")
    print("Saving Model Components...")
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save LoRA Adapter
    student_model.language_model.save_pretrained(output_dir)
    
    # 2. Save Projector Weights
    projector_path = os.path.join(output_dir, "projector.pt")
    torch.save(student_model.projector.state_dict(), projector_path)
    
    # 3. Save Tokenizer & Processor
    tokenizer.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    
    print(f"Model components saved to {output_dir}")

if __name__ == "__main__":
    train()
