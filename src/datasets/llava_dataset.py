import json
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

class LLaVADataset(Dataset):
    """
    Dataset for LLaVA distillation.
    Handles loading image-text pairs from LLaVA-style JSON data.
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: PreTrainedTokenizer,
        image_processor: ProcessorMixin,
        max_length: int = 512,
    ):
        """
        Args:
            data_path: Path to the LLaVA JSON file.
            image_folder: Path to the folder containing images.
            tokenizer: Tokenizer for the student language model (Qwen).
            image_processor: Image processor for the student vision model (SigLIP).
            max_length: Maximum sequence length for tokenization.
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load Image
        image_file = item.get("image")
        if image_file:
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a dummy image or handle error appropriately
                raise e
        else:
            raise ValueError(f"No image found for item {idx}")

        # Process Text
        # LLaVA format: "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        conversations = item.get("conversations", [])
        
        # Construct prompt for Qwen

        input_text = ""
        target_text = ""
        
        for turn in conversations:
            if turn["from"] == "human":
                input_text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif turn["from"] == "gpt":
                target_text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
                
        full_text = input_text + target_text
        
        # Tokenize
        # We need labels for training. 
        # For distillation, we might just need the full input_ids and attention_mask
        # The trainer will handle masking user tokens if using standard CLM loss, 
        # but for distillation we might want the student to generate the assistant response.
        
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length", # Pad here or in collator? Collator is better usually, but let's stick to simple for now.
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": input_ids.clone() # Standard CLM labels, will be masked in collator or trainer if needed
        }
