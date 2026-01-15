# Pre-Computed Feature Caching for Faster Training
import os
import sys
import torch
import yaml
import gc
import json
from transformers import AutoProcessor, SiglipVisionModel, AutoTokenizer
from tqdm import tqdm
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets.llava_dataset import LLaVADataset

# MICRO-SHARD: Only 1000 samples per shard (~600MB memory spike)
SHARD_SIZE = 1000

def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def precompute():
    with open("configs/qlora_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup paths
    save_dir = os.path.join(config["output_dir"], "cached_shards")
    progress_file = os.path.join(save_dir, "progress.json")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load Vision Model
    print("Loading Vision Model...")
    vision_tower = SiglipVisionModel.from_pretrained(
        config["vision_tower"], 
        torch_dtype=torch.float16  # Use FP16 to save memory
    )
    vision_tower.to(device)
    vision_tower.eval()
    
    # Freeze and disable gradients entirely
    for param in vision_tower.parameters():
        param.requires_grad = False

    # 2. Load Dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoProcessor.from_pretrained(config["vision_tower"], use_fast=True)

    print("Loading Dataset...")
    raw_dataset = LLaVADataset(
        data_path=config["data_path"],
        image_folder=config["image_folder"],
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    total_samples = len(raw_dataset)

    # 3. Resume Logic
    start_index = 0
    shard_idx = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
            start_index = progress.get("next_sample", 0)
            shard_idx = progress.get("next_shard", 0)
        print(f"📂 Resuming from sample {start_index} (shard {shard_idx})")
    
    if start_index >= total_samples:
        print("✅ All samples already processed!")
        return

    print(f"Pre-computing features for {total_samples - start_index} remaining samples...")
    print(f"Shard size: {SHARD_SIZE} | Saving to: {save_dir}")

    # 4. Processing Loop
    data_list = []
    
    with torch.no_grad():
        for i in tqdm(range(start_index, total_samples), initial=start_index, total=total_samples):
            try:
                sample = raw_dataset[i]
            except Exception as e:
                print(f"\n⚠️ Skipping sample {i}: {e}")
                continue
            
            # Forward pass
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device, dtype=torch.float16)
            vision_outputs = vision_tower(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state.cpu().squeeze(0).to(torch.float16)
            
            data_list.append({
                "input_ids": sample["input_ids"].cpu(),
                "labels": sample["labels"].cpu(),
                "attention_mask": sample["attention_mask"].cpu(),
                "image_features": image_features
            })
            
            # Save shard + clear memory
            if len(data_list) >= SHARD_SIZE:
                shard_path = os.path.join(save_dir, f"shard_{shard_idx:04d}")
                
                # Save shard
                hf_dataset = Dataset.from_list(data_list)
                hf_dataset.save_to_disk(shard_path)
                
                # Save progress for resume
                with open(progress_file, "w") as f:
                    json.dump({"next_sample": i + 1, "next_shard": shard_idx + 1}, f)
                
                # Clear memory aggressively
                del data_list, hf_dataset
                clear_memory()
                data_list = []
                shard_idx += 1
                
                print(f"\n💾 Saved shard {shard_idx - 1}")
    
    # Save remaining
    if len(data_list) > 0:
        shard_path = os.path.join(save_dir, f"shard_{shard_idx:04d}")
        hf_dataset = Dataset.from_list(data_list)
        hf_dataset.save_to_disk(shard_path)
        
        with open(progress_file, "w") as f:
            json.dump({"next_sample": total_samples, "next_shard": shard_idx + 1}, f)
        
        del data_list, hf_dataset
        clear_memory()
    
    print(f"\n✅ Done! Saved {shard_idx + 1} shards to {save_dir}")

if __name__ == "__main__":
    precompute()