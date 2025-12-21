import os
import sys
import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent

def export_model(adapter_path, output_path):
    # adapter_path should contain the LoRA adapter and projector.pt
    
    print(f"Loading adapter config from {adapter_path}")
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    vision_model_name = "google/siglip-base-patch16-224" # Hardcoded for now
    
    print(f"Base Model: {base_model_name}")
    
    print("Initializing Base Student...")
    # Load base student model (FP16)
    student_model = BonsaiStudent(
        vision_model_name=vision_model_name,
        language_model_name=base_model_name,
        load_in_4bit=False 
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    student_model.language_model = PeftModel.from_pretrained(
        student_model.language_model,
        adapter_path
    )
    
    print("Merging adapters...")
    student_model.language_model = student_model.language_model.merge_and_unload()
    
    print("Loading trained projector weights...")
    proj_path = os.path.join(adapter_path, "projector.pt")
    if os.path.exists(proj_path):
        projector_state = torch.load(proj_path, map_location="cpu")
        student_model.projector.load_state_dict(projector_state)
    else:
        print("WARNING: No projector.pt found! Exported model will have random projector weights.")
    
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Language Model (merged)
    lm_path = os.path.join(output_path, "language_model")
    student_model.language_model.save_pretrained(lm_path)
    
    # 2. Projector
    proj_out_path = os.path.join(output_path, "projector.pt")
    torch.save(student_model.projector.state_dict(), proj_out_path)
    
    # 3. Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(adapter_path) # Should be saved in adapter_path
    tokenizer.save_pretrained(lm_path)
    
    try:
        processor = AutoProcessor.from_pretrained(adapter_path)
        processor.save_pretrained(output_path)
    except:
        print("Processor not found in adapter path, loading from hub")
        processor = AutoProcessor.from_pretrained(vision_model_name)
        processor.save_pretrained(output_path)
    
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()
    
    export_model(args.adapter_path, args.output_path)
