import hashlib
import json
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

from src.utils.constants import IGNORE_INDEX, SYSTEM_PROMPT


class LLaVADataset(Dataset):
    """
    Dataset for LLaVA-style instruction tuning.

    Produces variable-length, answer-only-supervised samples:
      - Multi-turn conversations keep their real order (u1 a1 u2 a2 ...).
      - Prompt turns, chat-template scaffolding and the system turn are masked
        out of the loss with IGNORE_INDEX.
      - No padding here; the collator pads each batch to its own longest sample.
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: PreTrainedTokenizer,
        image_processor: ProcessorMixin,
        max_length: int = 640,
        max_samples: Optional[int] = None,
        skip_missing_images: bool = True,
    ):
        """
        Args:
            data_path: Path to the LLaVA JSON file.
            image_folder: Path to the folder containing images.
            tokenizer: Tokenizer for the student language model (Qwen).
            image_processor: Image processor for the student vision model (SigLIP).
            max_length: Maximum text sequence length (image tokens are added on top).
            max_samples: Optionally keep only the first N samples (subset training).
            skip_missing_images: Drop records whose image file is absent instead of
                crashing mid-epoch.
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        with open(data_path, "r") as f:
            self.data = json.load(f)

        if skip_missing_images:
            present = [
                item for item in self.data
                if item.get("image")
                and os.path.exists(os.path.join(image_folder, item["image"]))
            ]
            dropped = len(self.data) - len(present)
            if dropped:
                print(f"LLaVADataset: skipping {dropped} records with missing images "
                      f"({len(present)} usable).")
            self.data = present

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self) -> List[int]:
        """
        Token length per sample, for length-grouped batching.

        Computed from text only and cached on disk: HF's LengthGroupedSampler
        otherwise derives lengths by iterating the dataset, which here would
        decode and preprocess every image before training could start.
        """
        if getattr(self, "_lengths", None) is not None:
            return self._lengths

        fingerprint = hashlib.md5(
            f"{os.path.abspath(self.data_path)}|{len(self.data)}|{self.max_length}".encode()
        ).hexdigest()[:12]
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(self.data_path)), f".lengths_{fingerprint}.json"
        )

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._lengths = json.load(f)
            return self._lengths

        print(f"Computing sample lengths for length-grouped batching "
              f"({len(self.data)} samples, text only)...")
        lengths = []
        for item in self.data:
            messages = self._build_messages(item.get("conversations", []))
            n = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
            lengths.append(min(n, self.max_length))
        try:
            with open(cache_path, "w") as f:
                json.dump(lengths, f)
        except OSError:
            pass  # cache is an optimization, not a requirement
        self._lengths = lengths
        return self._lengths

    def _build_messages(self, conversations: List[Dict]) -> List[Dict[str, str]]:
        """Convert LLaVA 'conversations' into ordered chat messages."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in conversations:
            # The image is injected as prepended embeddings, so the textual
            # placeholder is removed here.
            text = turn["value"].replace("<image>", "").strip()
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": text})
        return messages

    def _tokenize_with_answer_only_labels(self, messages: List[Dict[str, str]]):
        """
        Tokenize the full conversation and supervise only assistant spans.

        Qwen's chat template is append-only, so the rendering of messages[:i] is a
        prefix of the rendering of messages[:i+1]. That lets us locate each
        assistant span by token-length difference rather than string matching.
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        labels = [IGNORE_INDEX] * len(input_ids)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            # Everything up to and including "<|im_start|>assistant\n" is context.
            prompt_len = len(self.tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True
            ))
            # Everything through this assistant turn's "<|im_end|>\n".
            answer_end = len(self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=True, add_generation_prompt=False
            ))
            for pos in range(prompt_len, min(answer_end, len(input_ids))):
                labels[pos] = input_ids[pos]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        return input_ids, labels

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        image_file = item.get("image")
        if not image_file:
            raise ValueError(f"No image found for item {idx}")

        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        messages = self._build_messages(item.get("conversations", []))
        input_ids, labels = self._tokenize_with_answer_only_labels(messages)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }
