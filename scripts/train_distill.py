import os
import sys
import yaml
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent
from src.arch.teacher_wrapper import LLaVATeacher
from src.training.qlora_trainer import BonsaiTrainer
from src.datasets.llava_dataset import LLaVADataset
from src.datasets.data_collator import DataCollatorForDistillation

def train():
    # Load Config
    with open("configs/qlora_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading Models...")
    # 1. Load Student
    student_model = BonsaiStudent(
        vision_model_name=config["vision_tower"],
        language_model_name=config["model_name_or_path"],
        load_in_4bit=True
    )
    
    # Enable Gradient Checkpointing
    student_model.language_model.gradient_checkpointing_enable()
    student_model.language_model = prepare_model_for_kbit_training(student_model.language_model)
    
    # 2. Apply LoRA to Student Language Model
    # Apply LoRA to the LM. The projector is trained fully. The vision tower is frozen.
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model.language_model = get_peft_model(student_model.language_model, peft_config)
    student_model.print_trainable_parameters() # Should show LoRA + Projector params
    
    # Ensure Projector is trainable
    for param in student_model.projector.parameters():
        param.requires_grad = True
        
    # 3. Load Teacher
    teacher_model = LLaVATeacher(model_id=config["teacher_model"])
    
    # 4. Load Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token # Qwen usually uses eos as pad or specific pad
    
    image_processor = AutoProcessor.from_pretrained(config["vision_tower"])
    
    # 5. Load Dataset
    dataset = LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        bf16=config["bf16"],
        tf32=config["tf32"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        dataloader_num_workers=config["dataloader_num_workers"],
        report_to=config["report_to"],
        remove_unused_columns=False # Important for custom models/datasets
    )
    
    # 7. Trainer
    trainer = BonsaiTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForDistillation(tokenizer=tokenizer),
        alpha=config["distill_alpha"],
        temperature=config["distill_temperature"]
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model(config["output_dir"])

if __name__ == "__main__":
    train()
