import os

import gradio as gr
import torch
from scripts.evaluate import build_prompt_ids, load_merged_model

# Global model loader
MODEL_PATH = "./checkpoints/bonsai-llava-v1-merged" # Default path, change as needed
# We load lazily or globally
model, processor, tokenizer = None, None, None

def init_model():
    global model, processor, tokenizer
    if model is None:
        # Check if path exists, else warn
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and export first.")
        model, processor, tokenizer = load_merged_model(MODEL_PATH)

def generate_response(image, prompt):
    if image is None:
        return "Please upload an image."
    
    init_model()
    
    # Preprocess. generate() casts pixels to the vision tower's dtype itself.
    device = model.language_model.device
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    input_ids = build_prompt_ids(tokenizer, prompt).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response

with gr.Blocks(title="Bonsai-LLaVA Demo") as demo:
    gr.Markdown("# Bonsai-LLaVA: Tiny Multimodal Model")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            text_input = gr.Textbox(label="Prompt", value="Describe this image.")
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            output_text = gr.Textbox(label="Response")
            
    submit_btn.click(
        fn=generate_response,
        inputs=[image_input, text_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
