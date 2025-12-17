import sys
import os
import pytest
import torch
import json
from PIL import Image
from unittest.mock import MagicMock

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.llava_dataset import LLaVADataset
from src.datasets.data_collator import DataCollatorForDistillation

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    processor.return_value = MagicMock(pixel_values=torch.randn(1, 3, 224, 224))
    return processor

@pytest.fixture
def sample_data(tmp_path):
    # Create a dummy image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test.jpg"
    Image.new('RGB', (100, 100)).save(img_path)
    
    # Create a dummy json
    json_path = tmp_path / "data.json"
    data = [
        {
            "image": "test.jpg",
            "conversations": [
                {"from": "human", "value": "What is this?"},
                {"from": "gpt", "value": "It is a test."}
            ]
        }
    ]
    with open(json_path, "w") as f:
        json.dump(data, f)
        
    return str(json_path), str(img_dir)

def test_dataset_loading(sample_data, mock_tokenizer, mock_image_processor):
    json_path, img_dir = sample_data
    dataset = LLaVADataset(json_path, img_dir, mock_tokenizer, mock_image_processor)
    
    assert len(dataset) == 1
    item = dataset[0]
    
    assert "input_ids" in item
    assert "pixel_values" in item
    assert item["pixel_values"].shape == (3, 224, 224)

def test_collator(mock_tokenizer):
    collator = DataCollatorForDistillation(mock_tokenizer)
    
    features = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor([1, 2])
        },
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor([1, 2, 3])
        }
    ]
    
    batch = collator(features)
    
    assert batch["input_ids"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    # Check padding (pad token is 0)
    assert batch["input_ids"][0, 2] == 0 
