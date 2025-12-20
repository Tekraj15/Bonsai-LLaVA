# Bonsai-LLaVA: Distilling & Quantizing Multimodal Giants

> **Bonsai-Llava** project is started with a sole purpose to practice the **Art of miniaturization without losing the Essence**.

> Bonsai-LLaVA is an experimental project aimed to distill the visual reasoning capabilities of **LLaVA-v1.5-7B** into a compact **<0.5B param(490M) student** (TinyLlama + SigLIP) model, optimized for consumer hardware and edge inference image-to-text generation. The implementation started with extreme architectural compression via Supervised Fine-Tuning (SFT) in Phase-I, with future plan to perform Hidden State Knowledge Distillation (HSKD) in Phase-II, to democratize Multimodal AI to consumer and edge hardware.

## Introduction
The current state of Vision-Language Models (VLMs) is dominated by massive parameters (7B+), making them inaccessible for real-time edge applications or resource-constrained environments. While standard quantization helps, it often degrades semantic reasoning.

**Bonsai-LLaVA** aims to solve this by combining **Knowledge Distillation** with **QLoRA**. We do not just shrink the weights; we teach a tiny, 4-bit quantized "Student" model to mimic the reasoning process of a "Teacher" giant. By fusing the efficiency of **Qwen2.5-0.5B** with the sharp visual encoding of **SigLIP-Base**, we create a VLM that fits on a Raspberry Pi 5 or local laptop while retaining strong reasoning capabilities.

**The Distillation Challenge**: Ideally, I would have started with standard Knowledge Distillation ($L_{KL}$) to teach the student. But, since the Teacher is LLaVA (Llama based), Student is Qwen, they have different tokenizers(vocabularies) which creates a  critical vocabulary mismatch (32k vs 152k) between teacher/student architectures. Standard logit-based distillation requires a complex alignment algorithm that is computationally expensive.  So, due to compute resource constraints and complexities of compressing models with different architectures, I have divided the project into 2 disttinct phases:


# Phase 1: Bonsai-LLaVA-SFT (Current)
"Architectural Compression & Efficient Transfer"

Focus: Establishing the "Pico" architecture (SigLIP + Qwen-0.5B) and transferring capabilities via Supervised Fine-Tuning (SFT), aiming to optimize the physical architecture and learning directly from high-quality instruction data.

Status: Implementation & Optimization in progress for Apple Silicon (MPS).

# Phase 2: Bonsai-LLaVA-HSD (Future work)
"Hidden State Knowledge Distillation"

Focus: enhancing the student's reasoning by aligning its internal "brain activity" with the teacher model's hidden states. This bypasses the vocabulary mismatch problem entirely by distilling "concepts" rather than "words."

Status: In Research / Planning.


## Mission
To democratize access to high-performance multimodal AI by proving that **architecture-aware distillation** combined with **4-bit Quantization (QLoRA)** can yield edge-native models that rival 7B counterparts while consuming **<2GB VRAM**.

---

## Objectives
### Phase I
1. **Rebuilt the LLaVA architecture from scratch**: Using Architectural Distillation via Transfer Learning, achieve a parameter count of <0.5B.
2. **Hardware-Aware Training**: Engineer a training pipeline optimized for Apple Silicon (MPS) using FP16 precision to avoid quantization instabilities.
3. **Edge Compatibility**:Ensure the Supervised fine-tuned student model runs on devices with <2GB VRAM (e.g., Raspberry Pi 5, Mobile Phones, Apple Silicon)

### Phase II
1. **Feature Alignment**: Implement Hidden State Distillation(HSD) to map the Student's intermediate layer activations to the Teacher's layers.
2. **MSE Loss Optimization**: Replace standard Cross-Entropy training with a composite loss ($L_{CE} + L_{MSE}$) to force the student to mimic the teacher's internal representation.
3. **Inference Acceleration**: Native integration with FlashAttention-2 for sub-20ms latency.
---

## Methodology: 

The core innovation of Bonsai-LLaVA is the **"Pico" Architecture**, which serves as the foundation for both phases.

1. **The Common Architecture ("Pico")**
We replace the heavy legacy components of Standard LLaVA with highly efficient alternatives:

### Phase I: The "Pico" Architecture (Architecture Search)

We rebuilt the LLaVA architecture from scratch, swapping heavy legacy components for modern, lightweight alternatives designed for efficiency.


2. **Phase I Training Strategy: "Efficient SFT"**
In Phase I, we rely on **Data-Driven Transfer**:
**Quantization**: The Qwen backbone is loaded in 4-bit NF4 precision (via QLoRA) and frozen.
**Optimization**: We train only the Projector and LoRA Adapters.
**Loss Function**: Standard Cross-Entropy ($L_{CE}$) on the LLaVA-Instruct-150k dataset.
**Why SFT?** It avoids the "Vocab Mismatch" issue entirely and is computationally cheaper, allowing us to validate the architecture's capabilities immediately.


3. **Phase II Training Strategy: "Hidden State Distillation"**
In Phase II, we will upgrade the training loop to perform **Feature-Level Distillation**:
Instead of matching output words (logits), we match internal vectors.
Method: A learnable "Alignment Projector" maps the Student's hidden states ($d=896$) to the Teacher's hidden states ($d=4096$).
**Loss Function**:
$$L_{total} = (1 - \alpha) L_{CE} + \alpha L_{MSE}(H_{Teacher}, Proj(H_{Student}))$$
Where $L_{MSE}$ minimizes the distance between the Student's "thought vector" and the Teacher's "thought vector."
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
