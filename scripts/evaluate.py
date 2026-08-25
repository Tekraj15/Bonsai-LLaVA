# CLI for testing the model on images.
import os
import sys
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arch.student_model import BonsaiStudent, BonsaiProjector
from src.utils.constants import SYSTEM_PROMPT
from src.utils.mps_compat import generation_device


def build_prompt_ids(tokenizer, prompt: str):
    """
    Render the prompt exactly as training did.

    Training formats every sample with apply_chat_template and SYSTEM_PROMPT, so
    inference must use the same call — a hand-written prompt string here would
    feed the model a prefix it never saw.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

DEFAULT_VISION_TOWER = "google/siglip-base-patch16-224"


def load_merged_model(model_path):
    """
    Reconstruct a BonsaiStudent from either layout produced by this repo:

      - a TRAINING output   — adapter_config.json + projector.pt (LoRA applied on
        top of the base LLM), or
      - an EXPORT output    — language_model/ holding a merged model + projector.pt.

    Detecting the layout rather than assuming one is what keeps train -> export ->
    evaluate composable; the previous version assumed the adapter layout and so
    could not load anything export_merged.py wrote.
    """
    print(f"Loading model from {model_path}...")

    merged_lm_dir = os.path.join(model_path, "language_model")
    is_merged = os.path.isdir(merged_lm_dir)
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if not (is_merged or is_adapter):
        raise FileNotFoundError(
            f"{model_path} contains neither adapter_config.json (training output) "
            f"nor language_model/ (export output). Nothing to load."
        )

    vision_model_name = DEFAULT_VISION_TOWER
    if is_merged:
        base_llm_name = merged_lm_dir
        print(f"Detected merged export. Base LLM: {base_llm_name}")
    else:
        from peft import PeftConfig
        base_llm_name = PeftConfig.from_pretrained(model_path).base_model_name_or_path
        print(f"Detected LoRA adapter. Base LLM: {base_llm_name}")
    print(f"Vision Tower: {vision_model_name}")

    # Initialize Student (FP16 for evaluation)
    model = BonsaiStudent(
        vision_model_name=vision_model_name,
        language_model_name=base_llm_name,
        load_in_4bit=False
    )

    if is_adapter:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, model_path)

    # 4. Load Projector
    proj_path = os.path.join(model_path, "projector.pt")
    if os.path.exists(proj_path):
        print(f"Loading projector from {proj_path}")
        projector_state = torch.load(proj_path, map_location="cpu")
        model.projector.load_state_dict(projector_state)
    else:
        print("WARNING: No projector.pt found! Using random projector weights.")
        
    # 5. Load Processor & Tokenizer, preferring the copies saved beside the model.
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(vision_model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # Merged exports save the tokenizer next to the language model.
        tokenizer = AutoTokenizer.from_pretrained(base_llm_name)

    # Move to device. On macOS 13 this resolves to CPU: MPS aborts inside
    # generate() with a Metal matmul shape error (see src/utils/mps_compat.py).
    device = generation_device()
    print(f"Moving model to {device} for generation...")
    model.to(device)
    model.language_model.to(device)
    model.vision_tower.to(device)
    model.projector.to(device)
    model.device = device
    model.eval()
    
    return model, processor, tokenizer

def evaluate(model_path, image_path, prompt="Describe this image.",
             max_new_tokens=100, do_sample=False, temperature=0.7):
    model, processor, tokenizer = load_merged_model(model_path)

    image = Image.open(image_path).convert("RGB")

    # Prepare inputs. forward()/generate() cast pixels to the vision tower's own
    # dtype, so no manual FP16 cast is needed (and a wrong one crashes conv2d).
    device = model.language_model.device
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    input_ids = build_prompt_ids(tokenizer, prompt).to(device)

    print("Generating...")
    with torch.no_grad():
        # Greedy by default: benchmark numbers must be reproducible. Pass
        # --sample for the exploratory/demo behaviour.
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Response: {response}")
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--sample", action="store_true",
                        help="Sample instead of greedy decoding (demo only; "
                             "benchmarks must stay greedy to be reproducible).")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    evaluate(args.model_path, args.image_path, args.prompt,
             max_new_tokens=args.max_new_tokens, do_sample=args.sample,
             temperature=args.temperature)
