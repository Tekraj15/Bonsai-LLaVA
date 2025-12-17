import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import PreTrainedTokenizer

@dataclass
class DataCollatorForDistillation:
    """
    Data collator for LLaVA distillation.
    Pads input_ids, attention_mask, and pixel_values.
    """
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad text
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100 # Ignore index for loss
        )

        # Stack images
        pixel_values = torch.stack(pixel_values)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }
