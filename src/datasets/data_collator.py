import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from transformers import PreTrainedTokenizer

from src.utils.constants import IGNORE_INDEX


@dataclass
class CustomeDataCollatorForLlava:
    """
    Data collator for LLaVA-style SFT.

    Pads each batch to its own longest sample (never to a fixed max length) and
    handles either pixel_values (raw images) or image_features (pre-computed
    embeddings). Label padding uses IGNORE_INDEX so padded positions contribute
    no loss.
    """
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id. Set a real pad token — do not "
                "alias it to eos, or padding becomes indistinguishable from "
                "end-of-turn."
            )

    def _pad_to_multiple(self, tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        if not self.pad_to_multiple_of:
            return tensor
        seq_len = tensor.shape[1]
        remainder = seq_len % self.pad_to_multiple_of
        if remainder == 0:
            return tensor
        pad_len = self.pad_to_multiple_of - remainder
        padding = torch.full(
            (tensor.shape[0], pad_len), pad_value,
            dtype=tensor.dtype, device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=1)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 1. Collect Text Data
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 2. Pad Text Data to the longest sample in THIS batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = self._pad_to_multiple(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self._pad_to_multiple(attention_mask, 0)
        labels = self._pad_to_multiple(labels, IGNORE_INDEX)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 3. Handle Visual Data (Conditional)
        if "image_features" in features[0]:
            # Case A: Cached Features (Fast Training)
            batch["image_features"] = torch.stack([f["image_features"] for f in features])
        elif "pixel_values" in features[0]:
            # Case B: Raw Images (Inference / Pre-compute)
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

        return batch
