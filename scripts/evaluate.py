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
    lm_path = os.path.join(model_path, "language_model")
    proj_path = os.path.join(model_path, "projector.pt")
    
    print("Loading merged model components...")
    # 1. Load LM
    language_model = AutoModelForCausalLM.from_pretrained(lm_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(lm_path)
    
    # 2. Load Vision (SigLIP base)
    # Ideally it should be saved in config. Hardcoding default for now or pass arg.
    vision_model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_path) # Saved in root of output
    
    # 3. Initialize Student
    # Passing load_in_4bit=False because we are loading a merged FP16 model
    model = BonsaiStudent(
        vision_model_name=vision_model_name,
        language_model_name=lm_path, # Pass path to load local LM
        load_in_4bit=False
    )
    
    print("Loading projector weights...")
    projector_state = torch.load(proj_path)
    model.projector.load_state_dict(projector_state)
    
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
