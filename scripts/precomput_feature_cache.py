import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
from datasets import Dataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets.llava_dataset import LLaVADataset

def precompute():
    # Load Config
    with open("configs/qlora_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Vision Model (SigLIP Vision Encoder ONLY - not the full multimodal model)
    print("Loading Vision Model...")
    vision_tower = SiglipVisionModel.from_pretrained(config["vision_tower"])
    vision_tower.to(device)
    vision_tower.eval()

    # 2. Load Processor & Tokenizer
    # We use the raw dataset class to handle image loading
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = AutoProcessor.from_pretrained(config["vision_tower"])

    print("Loading Dataset...")
    raw_dataset = LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # 3. Processing Loop
    # We will store: input_ids, labels, attention_mask, and image_features (tensors)
    data_list = []
    
    print(f"Pre-computing features for {len(raw_dataset)} samples...")
    
    # We iterate one by one (or small batch) to save memory
    with torch.no_grad():
        for i in tqdm(range(len(raw_dataset))):
            sample = raw_dataset[i]
            
            # Move pixel_values to GPU for inference
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device) # Add batch dim
            
            # Forward Pass (Vision Encoder Only)
            vision_outputs = vision_tower(pixel_values=pixel_values)
            # Get the features: shape [1, 196, 768]
            image_features = vision_outputs.last_hidden_state.cpu().squeeze(0) # Move back to CPU to save
            
            # Store everything needed for training
            data_list.append({
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "attention_mask": sample["attention_mask"],
                "image_features": image_features # The heavy lifting is done!
            })

    # 4. Save as a Hugging Face Dataset (Efficient Disk Format)
    print("Saving Cached Dataset...")
    hf_dataset = Dataset.from_list(data_list)
    
    save_path = os.path.join(config["output_dir"], "cached_dataset")
    hf_dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    precompute()