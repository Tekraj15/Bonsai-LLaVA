# The LLaVA-7B wrapper.
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LLaVATeacher(nn.Module):
    """
    Wrapper for LLaVA-v1.5-7B Teacher Model.
    Optimized for inference (eval mode, no gradients).
    """
    def __init__(self, model_id: str = "liuhaotian/llava-v1.5-7b"):
        super().__init__()
        print(f"Loading Teacher Model: {model_id}")
        
        # Load in 4-bit or 8-bit to save memory if possible, or FP16
        # Ideally teacher is FP16 for best logits, but 7B FP16 is 14GB.
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
            # load_in_4bit=True # Optional: Enable if VRAM is tight
        )
        self.model.eval()
        self.model.requires_grad_(False)
        
    def forward(self, input_ids, pixel_values, attention_mask=None):
        """
        Get logits from teacher.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask
            )
        return outputs.logits
