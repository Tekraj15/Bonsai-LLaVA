import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from transformers import PreTrainedTokenizer

@dataclass
class CustomeDataCollatorForLlava:
    """
    CustomData collator for LLaVA(For SFT now) .
    Pads input_ids, attention_mask, and handles either pixel_values (raw images) 
    or image_features (pre-computed embeddings).
    """
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 1. Collect Text Data
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 2. Pad Text Data
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100 # Ignore index for loss
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        # 3. Handle Visual Data (Conditional)
        # Check the first sample to see what kind of data we have
        if "image_features" in features[0]:
            # Case A: Cached Features (Fast Training)
            # Stack list of tensors [196, 768] -> [Batch, 196, 768]
            batch["image_features"] = torch.stack([f["image_features"] for f in features])
            
        elif "pixel_values" in features[0]:
            # Case B: Raw Images (Inference / Pre-compute)
            # Stack list of tensors [3, 224, 224] -> [Batch, 3, 224, 224]
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

        return batch