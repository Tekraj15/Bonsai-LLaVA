import os
import sys
import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent

def export_model(adapter_path, output_path, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", vision_model_name="google/siglip-base-patch16-224"):
    print(f"Loading base model: {base_model_name}")
    # Load base student model (without 4-bit for merging, usually FP16)
    student_model = BonsaiStudent(
        vision_model_name=vision_model_name,
        language_model_name=base_model_name,
        load_in_4bit=False # Load in FP16 for merging
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    
    # Load LoRA adapters
    # BonsaiStudent wraps the LM in .language_model, so applying PEFT to that specific submodule
    student_model.language_model = PeftModel.from_pretrained(
        student_model.language_model,
        adapter_path
    )
    
    print("Merging adapters...")
    student_model.language_model = student_model.language_model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # Save the full student model(LM and the Projector) in components(separately)
    # Coz BonsaiStudent is a custom class, we can't just use .save_pretrained() on the whole thing easily.
    
    # 1. Language Model (merged)
    lm_path = os.path.join(output_path, "language_model")
    student_model.language_model.save_pretrained(lm_path)
    
    # 2. Projector
    proj_path = os.path.join(output_path, "projector.pt")
    torch.save(student_model.projector.state_dict(), proj_path)
    
    # 3. Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(lm_path)
    
    processor = AutoProcessor.from_pretrained(vision_model_name)
    processor.save_pretrained(output_path)
    
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()
    
    export_model(args.adapter_path, args.output_path)
