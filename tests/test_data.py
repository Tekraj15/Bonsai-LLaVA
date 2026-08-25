"""
Data-contract tests.

These target the failure modes that silently invalidated earlier training runs:
loss on padding, loss on prompts, scrambled multi-turn order, pad aliased to eos,
and static max-length padding. They use the real Qwen tokenizer (cached locally)
because every one of these bugs lived in the interaction with the real chat
template — a mocked tokenizer would have passed while the pipeline was broken.
"""
import json
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.data_collator import CustomeDataCollatorForLlava
from src.datasets.llava_dataset import LLaVADataset
from src.utils.constants import IGNORE_INDEX

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VISION = "google/siglip-base-patch16-224"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, use_fast=True)


@pytest.fixture(scope="module")
def image_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(VISION, use_fast=True)


@pytest.fixture
def multi_turn_data(tmp_path):
    """A 3-round conversation — the case the old pipeline scrambled."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (120, 90), color=(30, 90, 140)).save(img_dir / "img0.jpg")

    records = [{
        "id": "img0",
        "image": "img0.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat colour is the bus?"},
            {"from": "gpt", "value": "The bus is white and red."},
            {"from": "human", "value": "What is on its back?"},
            {"from": "gpt", "value": "An advertisement."},
            {"from": "human", "value": "Is it moving?"},
            {"from": "gpt", "value": "Yes, it is driving down the street."},
        ],
    }]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(records))
    return str(json_path), str(img_dir)


@pytest.fixture
def dataset(multi_turn_data, tokenizer, image_processor):
    json_path, img_dir = multi_turn_data
    return LLaVADataset(json_path, img_dir, tokenizer, image_processor, max_length=640)


# --- A4: pad must not be aliased to eos -------------------------------------

def test_pad_token_is_distinct_from_eos(tokenizer):
    assert tokenizer.pad_token_id is not None
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "pad aliased to eos makes padding indistinguishable from end-of-turn"
    )


# --- A2: multi-turn ordering ------------------------------------------------

def test_turns_are_interleaved_in_order(dataset, tokenizer):
    text = tokenizer.decode(dataset[0]["input_ids"])
    positions = [
        text.index("What colour is the bus?"),
        text.index("The bus is white and red."),
        text.index("What is on its back?"),
        text.index("An advertisement."),
        text.index("Is it moving?"),
        text.index("Yes, it is driving down the street."),
    ]
    assert positions == sorted(positions), (
        "conversation turns are not interleaved u1 a1 u2 a2 ... — "
        "questions and answers were grouped separately"
    )


# --- A1/A3: answer-only supervision -----------------------------------------

def test_only_assistant_spans_are_supervised(dataset, tokenizer):
    item = dataset[0]
    supervised = tokenizer.decode(
        [t for t in item["labels"].tolist() if t != IGNORE_INDEX]
    )
    for answer in ["The bus is white and red.", "An advertisement.",
                   "Yes, it is driving down the street."]:
        assert answer in supervised

    for prompt_text in ["What colour is the bus?", "What is on its back?",
                        "Is it moving?", "helpful visual assistant"]:
        assert prompt_text not in supervised, (
            f"{prompt_text!r} is being supervised; prompts and the system turn "
            "must be masked"
        )


def test_eos_is_supervised(dataset, tokenizer):
    """The model must learn to STOP, so <|im_end|> is part of the target."""
    supervised = tokenizer.decode(
        [t for t in dataset[0]["labels"].tolist() if t != IGNORE_INDEX]
    )
    assert tokenizer.eos_token in supervised


def test_labels_align_with_input_ids(dataset):
    item = dataset[0]
    labels, input_ids = item["labels"], item["input_ids"]
    assert labels.shape == input_ids.shape
    mask = labels != IGNORE_INDEX
    assert torch.equal(labels[mask], input_ids[mask]), (
        "supervised labels must be the corresponding input token ids"
    )


def test_supervised_fraction_is_substantial(dataset):
    labels = dataset[0]["labels"]
    fraction = float((labels != IGNORE_INDEX).sum()) / labels.numel()
    assert 0.2 < fraction < 0.95, (
        f"supervised fraction {fraction:.1%} is implausible — near 0 means "
        "nothing is trained, near 1.0 means prompts leaked into the loss"
    )


# --- A5: no static padding --------------------------------------------------

def test_sequences_are_not_padded_to_max_length(dataset):
    item = dataset[0]
    assert len(item["input_ids"]) < 640, (
        "sample was padded to max_length; padding belongs in the collator"
    )
    assert item["attention_mask"].sum() == len(item["attention_mask"]), (
        "an unpadded sample must have an all-ones attention mask"
    )


# --- Collator contract ------------------------------------------------------

def test_collator_pads_dynamically_and_masks_padding(tokenizer):
    collator = CustomeDataCollatorForLlava(tokenizer=tokenizer, pad_to_multiple_of=None)
    features = [
        {"input_ids": torch.tensor([1, 2]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([IGNORE_INDEX, 2]),
         "pixel_values": torch.randn(3, 224, 224)},
        {"input_ids": torch.tensor([1, 2, 3, 4]),
         "attention_mask": torch.tensor([1, 1, 1, 1]),
         "labels": torch.tensor([IGNORE_INDEX, 2, 3, 4]),
         "pixel_values": torch.randn(3, 224, 224)},
    ]
    batch = collator(features)

    assert batch["input_ids"].shape == (2, 4), "batch should pad to its own longest"
    assert batch["input_ids"][0, 2] == tokenizer.pad_token_id
    assert batch["attention_mask"][0, 2] == 0
    assert batch["labels"][0, 2] == IGNORE_INDEX, "padded positions must not train"
    assert batch["pixel_values"].shape == (2, 3, 224, 224)


def test_collator_never_supervises_padded_positions(tokenizer):
    collator = CustomeDataCollatorForLlava(tokenizer=tokenizer)
    features = [
        {"input_ids": torch.arange(n), "attention_mask": torch.ones(n, dtype=torch.long),
         "labels": torch.arange(n), "pixel_values": torch.randn(3, 224, 224)}
        for n in (3, 9, 17)
    ]
    batch = collator(features)
    padded = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][padded] == IGNORE_INDEX)


def test_collator_handles_cached_features(tokenizer):
    collator = CustomeDataCollatorForLlava(tokenizer=tokenizer)
    features = [
        {"input_ids": torch.arange(5), "attention_mask": torch.ones(5, dtype=torch.long),
         "labels": torch.arange(5), "image_features": torch.randn(196, 768)}
        for _ in range(2)
    ]
    batch = collator(features)
    assert batch["image_features"].shape == (2, 196, 768)
    assert "pixel_values" not in batch


def test_collator_rejects_aliased_pad_token(tokenizer):
    import copy
    broken = copy.deepcopy(tokenizer)
    broken.pad_token = None
    with pytest.raises(ValueError):
        CustomeDataCollatorForLlava(tokenizer=broken)


# --- Robustness -------------------------------------------------------------

def test_missing_images_are_skipped_not_fatal(tmp_path, tokenizer, image_processor):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (64, 64)).save(img_dir / "present.jpg")

    records = [
        {"image": "present.jpg", "conversations": [
            {"from": "human", "value": "<image>\nHi"}, {"from": "gpt", "value": "Hello."}]},
        {"image": "absent.jpg", "conversations": [
            {"from": "human", "value": "<image>\nHi"}, {"from": "gpt", "value": "Hello."}]},
    ]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(records))

    dataset = LLaVADataset(str(json_path), str(img_dir), tokenizer, image_processor)
    assert len(dataset) == 1, "a missing image must not be able to kill an epoch"


def test_lengths_are_cheap_and_consistent(dataset):
    """lengths must not require decoding images (it runs before training)."""
    lengths = dataset.lengths
    assert len(lengths) == len(dataset)
    assert lengths[0] == len(dataset[0]["input_ids"])
