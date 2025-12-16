# Download data from HF Hub
import os
import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download

def download_data(output_dir):
    """
    Downloads the LLaVA-Instruct-150K dataset and COCO images.
    """
    print(f"Downloading LLaVA-Instruct-150K to {output_dir}...")
    
    # Create directories
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download JSONs
    # The dataset repository contains the json files
    try:
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        # Save as json for our loader
        json_path = os.path.join(raw_dir, "llava_instruct_150k.json")
        dataset.to_json(json_path)
        print(f"Saved instructions to {json_path}")
    except Exception as e:
        print(f"Error downloading instructions: {e}")
        print("Please manually download llava_instruct_150k.json from HuggingFace.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save data")
    args = parser.parse_args()
    
    download_data(args.output_dir)
