# Download data from HF Hub
import os
import argparse
from huggingface_hub import hf_hub_download
import shutil

def download_data(output_dir):
    """
    Downloads the LLaVA-Instruct-150K dataset and COCO images.
    """
    print(f"Downloading LLaVA-Instruct-150K to {output_dir}...")
    
    # Create directories
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download JSON directly using hf_hub_download
    # This avoids 'datasets' library processing errors and just gets the file we need
    try:
        print("Downloading llava_instruct_150k.json...")
        file_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
            local_dir=raw_dir,
            local_dir_use_symlinks=False # Download actual file
        )
        print(f"Successfully downloaded to {file_path}")
        
    except Exception as e:
        print(f"Error downloading instructions: {e}")
        print("Please manually download llava_instruct_150k.json from HuggingFace.")

    print("\nIMPORTANT: we also need the COCO 2017 Train images but that's huge in size")
    print("So, we need download 'train2017.zip' manually from http://images.cocodataset.org/zips/train2017.zip")
    print("And unzip it into {raw_dir}/images/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save data")
    args = parser.parse_args()
    
    download_data(args.output_dir)
