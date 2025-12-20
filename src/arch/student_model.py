# The Pico model (Qwen + SigLIP)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, PreTrainedModel, AutoConfig, SiglipVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple

class BonsaiProjector(nn.Module):
    """
    Projects visual features (SigLIP) to text embedding space (Qwen).
    """
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, x):
        return self.net(x)

class BonsaiStudent(nn.Module):
    """
    The Student Model: SigLIP Vision Encoder + Qwen Language Model.
    """
    def __init__(
        self, 
        vision_model_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit: bool = True
    ):
        super().__init__()
        
        # 1. Vision Encoder (SigLIP)
        print(f"Loading Vision Encoder: {vision_model_name}")
        self.vision_tower = SiglipVisionModel.from_pretrained(vision_model_name)
        self.vision_tower.requires_grad_(False) # Freeze vision tower usually
        
        # 2. Language Model (Qwen)
        print(f"Loading Language Model: {language_model_name}")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            # Fix for MPS BFloat16 on macOS < 14: Load config, set to float16, load on CPU first
            config = AutoConfig.from_pretrained(language_model_name)
            config.torch_dtype = torch.float16
            
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map={"":"cpu"} # Force CPU loading to handle BF16->FP16 cast safely
            )
            # We will move it to device in _initialize() or let Trainer handle it
        else:
            config = AutoConfig.from_pretrained(language_model_name)
            config.torch_dtype = torch.float16
            
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map={"":"cpu"}
            )
            
        # Freeze LM backbone for doing QLoRA (usually handled by PEFT, but good to be explicit if not using PEFT lib immediately)
        # For now, we assume the trainer will handle PEFT/LoRA injection.
        
        # 3. Projector
        vision_dim = self.vision_tower.config.hidden_size # 768 for SigLIP-Base
        text_dim = self.language_model.config.hidden_size # 896 for Qwen-0.5B
        
        self.projector = BonsaiProjector(vision_dim, text_dim)
        
        # Initialize device and move components
        self._initialize()
        
    def _initialize(self):
        """Initialize the model and processor."""
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            
        print(f"Selected device: {self.device}")
        
        if self.device == "cpu":
            print("--- Device Diagnostic ---")
            print(f"Torch version: {torch.__version__}")
            print(f"MPS available: {torch.backends.mps.is_available()}")
            print(f"MPS built: {torch.backends.mps.is_built()}")
            import platform
            print(f"Platform: {platform.platform()}")
            print(f"Processor: {platform.processor()}")
            print("-------------------------")
            
        print(f"Moving Vision Tower and Projector to {self.device}...")
        self.vision_tower.to(self.device)
        self.projector.to(self.device)
        
        # Explicitly move LM to device (since we loaded it on CPU)
        if hasattr(self, "language_model"):
            print(f"Moving Language Model to {self.device}...")
            self.language_model.to(self.device)
        
    def get_vision_tower(self):
        return self.vision_tower

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass for training/inference.
        We need to inject image embeddings into the input_embeddings of the LM.
        """
        
        # 1. Extract Image Features
        with torch.no_grad():
            vision_outputs = self.vision_tower(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state # [B, num_patches, vision_dim]
            
        # 2. Project to Text Space
        image_embeddings = self.projector(image_features) # [B, num_patches, text_dim]
        
        # 3. Embed Text
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 4. Multimodal Fusion
        # Simple approach for LLaVA for inserting images is to replace a special <image> token with the image embeddings. But handling this Dynamically in a batch is complex.
        # Prepend Strategy: Image Embeddings + Clean Text
        
        combined_embeds = torch.cat([image_embeddings, inputs_embeds], dim=1)
        
        # Update attention mask
        # image_embeddings shape: [B, I, D]
        # attention_mask shape: [B, T]
        batch_size = input_ids.shape[0]
        image_seq_len = image_embeddings.shape[1]
        
        if attention_mask is not None:
            image_mask = torch.ones((batch_size, image_seq_len), device=attention_mask.device)
            combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            combined_mask = None
            
        # Update labels if provided (shift them)
        if labels is not None:
            # We don't calculate loss on images, so pad labels with ignore_index
            ignore_pad = torch.full((batch_size, image_seq_len), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([ignore_pad, labels], dim=1)
        else:
            combined_labels = None
            
        # 5. LM Forward
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            return_dict=True,
            **kwargs
        )
        
        return outputs

    def generate(self, input_ids, pixel_values, **kwargs):
        """
        Custom generate method that handles image encoding and projection.
        """
        # Ensure components are on the correct device
        device = self.language_model.device
        self.vision_tower.to(device)
        self.projector.to(device)
        
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        
        # 1. Extract Image Features
        with torch.no_grad():
            vision_outputs = self.vision_tower(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
        
        # 2. Project to Text Space
        image_embeddings = self.projector(image_features)
        
        # 3. Embed Text
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 4. Multimodal Fusion (Prepend)
        combined_embeds = torch.cat([image_embeddings, inputs_embeds], dim=1)
        
        # 5. Generate
        return self.language_model.generate(inputs_embeds=combined_embeds, **kwargs)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
