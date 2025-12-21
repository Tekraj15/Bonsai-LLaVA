# CLI for testing the model on images.
import os
import sys
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent, BonsaiProjector

def load_merged_model(model_path):
    # Reconstruct BonsaiStudent from saved components
    # model_path should contain adapter_config.json, adapter_model.bin, projector.pt
    
    print(f"Loading model from {model_path}...")
    
    # 1. Load Config (to get base model names)
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(model_path)
    base_llm_name = peft_config.base_model_name_or_path
    
    # Hardcoded vision tower for now, or could be saved in a config
    vision_model_name = "google/siglip-base-patch16-224"
    
    print(f"Base LLM: {base_llm_name}")
    print(f"Vision Tower: {vision_model_name}")
    
    # 2. Initialize Student (Base)
    # Load it in FP16 for evaluation
    model = BonsaiStudent(
        vision_model_name=vision_model_name,
        language_model_name=base_llm_name,
        load_in_4bit=False
    )
    
    # 3. Load LoRA Adapter
    from peft import PeftModel
    model.language_model = PeftModel.from_pretrained(model.language_model, model_path)
    
    # 4. Load Projector
    proj_path = os.path.join(model_path, "projector.pt")
    if os.path.exists(proj_path):
        print(f"Loading projector from {proj_path}")
        projector_state = torch.load(proj_path, map_location="cpu")
        model.projector.load_state_dict(projector_state)
    else:
        print("WARNING: No projector.pt found! Using random projector weights.")
        
    # 5. Load Processor & Tokenizer
    processor = AutoProcessor.from_pretrained(vision_model_name) # Load from hub or local if saved
    # Try loading from model_path first
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except:
        pass
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model.to(device)
    model.eval()
    
    return model, processor, tokenizer

def evaluate(model_path, image_path, prompt="Describe this image."):
    model, processor, tokenizer = load_merged_model(model_path)
    
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.language_model.device).to(torch.float16)
    
    # Format prompt (Simple chat format)
    # <|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    # Model training assumed image embeddings prepended.
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.language_model.device)
    
    print("Generating...")
    with torch.no_grad():
        # Use the model's internal generate method
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Response: {response}")
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.image_path, args.prompt)
