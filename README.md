# Bonsai-LLaVA: Distilling & Quantizing Multimodal Giants

> **Bonsai-Llava** project is started with a sole purpose to practice the **Art of miniaturization without losing the Essence**.

> It's a research practice to distill the visual reasoning capabilities of **LLaVA-v1.5-7B** into a compact **<1B param(490M) student** (TinyLlama + SigLIP) model, optimized for consumer hardware and edge inference text-to-image generation.

## Introduction
The current state of Vision-Language Models (VLMs) is dominated by massive parameters (7B+), making them inaccessible for real-time edge applications or resource-constrained environments. While standard quantization helps, it often degrades semantic reasoning.

**Bonsai-LLaVA** solves this by combining **Knowledge Distillation** with **QLoRA**. We do not just shrink the weights; we teach a tiny, 4-bit quantized "Student" model to mimic the reasoning process of a "Teacher" giant. By fusing the efficiency of **Qwen2.5-0.5B** with the sharp visual encoding of **SigLIP-Base**, we create a VLM that fits on a Raspberry Pi 5 or local laptop while retaining strong reasoning capabilities.

## Mission
To democratize access to high-performance multimodal AI by proving that **architecture-aware distillation** combined with **4-bit Quantization (QLoRA)** can yield edge-native models that rival 7B counterparts while consuming **<2GB VRAM**.

---

## Objectives
1.  **Construct "Pico" Architecture:** Fuse `Qwen2.5-0.5B-Instruct` (Language) with `SigLIP-Base-Patch16-224` (Vision) using a custom MLP projector, keeping total parameters under **0.7B**.
2.  **Implement QLoRA Distillation:** Train the student using **Quantized Low-Rank Adaptation**. The student backbone is frozen in 4-bit NF4 precision, and only the adapters are trained to minimize the **KL-Divergence** between the Student's and Teacher's logits.
3.  **Optimize for Inference:** Native integration with **FlashAttention-2** and **vLLM** for production serving, targeting <20ms latency.

---

## Methodology: The "Shrink & Teach" Pipeline

Our pipeline follows a three-stage compression process designed to retain intelligence while slashing memory usage.

### 1. Architecture Alignment (The "Pico" Setup)
We replace the heavy encoders of standard LLaVA with highly efficient alternatives:
* **Vision:** Swapped CLIP-Large (300M) for **SigLIP-Base (200M)**. SigLIP's sigmoid loss function allows it to learn better fine-grained features with fewer parameters.
* **Language:** Swapped Vicuna-7B (7B) for **Qwen2.5-0.5B (0.5B)**.
* **Connector:** A trainable 2-layer MLP projects 768-dim visual features to the 896-dim text space of Qwen.

### 2. QLoRA Distillation (The Training)
Instead of standard fine-tuning (which requires massive VRAM), we use **Parameter-Efficient Distillation**:
* **Quantization:** The Qwen-0.5B backbone is loaded in **4-bit NF4** precision. Its weights are frozen.
* **Adaptation:** Low-Rank Adapters (LoRA) are injected into the attention and MLP layers.
* **Loss Function:** We train *only* the Projector and the LoRA adapters using a composite loss:
    $$L_{total} = (1 - \alpha) L_{CE} + \alpha T^2 L_{KL}(P_{Teacher} || P_{Student})$$
    * Where $L_{CE}$ learns the ground truth text.
    * Where $L_{KL}$ forces the student to match the Teacher's probability distribution (logits), effectively transferring the "reasoning style" of the 7B model.

### 3. Inference Optimization
The final model runs in a highly optimized state:
* **Memory:** ~600MB VRAM (vs 14GB for LLaVA-7B).
* **Speed:** Uses FlashAttention-2 kernels to eliminate memory bottlenecks during decoding.

---


## Tech Stack

### Core Components
* **Teacher Model:** `liuhaotian/llava-v1.5-7b` (The Industry Standard)
* **Student Language Backbone:** `Qwen/Qwen2.5-0.5B-Instruct` (State-of-the-art sub-1B model)
* **Student Vision Backbone:** `google/siglip-base-patch16-224` (Sigmoid Loss for efficient zero-shot localization)

### Optimization & Training
* **Framework:** PyTorch 2.1+, Hugging Face Transformers, `bitsandbytes`
* **Compression:** **QLoRA** (4-bit NormalFloat quantization + LoRA Adapters)
* **Distillation Logic:** Custom `Trainer` implementing Multi-Task Loss ($L_{Task} + \alpha L_{KL\_Div}$)
* **Acceleration:** FlashAttention-2, Gradient Checkpointing

### Inference Engine
* **Serving:** vLLM (Custom Model Registration)
* **Format:** SafeTensors (FP16 or INT4)

---


## 📊 Projected Benchmarks (Target)

| Metric | LLaVA-v1.5-7B | **Bonsai-LLaVA (Pico)** | Change |
| :--- | :--- | :--- | :--- |
| **Parameters** | 7.2 Billion | **~0.7 Billion** | 🔻 **90%** |
| **VRAM (Training)** | ~80GB (Full FT) | **<8GB (QLoRA)** | Edge Ready |
| **VRAM (Inference)** | ~14GB (FP16) | **~1GB (INT4)** | **14x Smaller** |
| **Architecture** | CLIP-L + LLaMA-2 | SigLIP-B + Qwen2.5 | Modern Stack |
