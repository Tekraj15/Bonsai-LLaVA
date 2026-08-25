# Bonsai-LLaVA: Distilling & Quantizing Multimodal Giants

> **Bonsai-Llava** project is started with a sole purpose to practice the **Art of miniaturization without losing the Essence**.

> Bonsai-LLaVA is an experimental project aimed to compress the visual reasoning capabilities of **LLaVA-v1.5-7B** into a compact **~0.59B param student** (Qwen2.5-0.5B-Instruct + SigLIP-Base), optimized for consumer hardware and edge inference image-to-text generation. The implementation started with extreme architectural compression via Supervised Fine-Tuning (SFT) in Phase-I, with future plan to perform Hidden State Knowledge Distillation (HSKD) in Phase-II, to democratize Multimodal AI to consumer and edge hardware.

## Introduction
The current state of Vision-Language Models (VLMs) is dominated by massive parameters (7B+), making them inaccessible for real-time edge applications or resource-constrained environments. While standard quantization helps, it often degrades semantic reasoning.

**Bonsai-LLaVA** aims to solve this by combining **architectural compression** with **parameter-efficient training** and **post-training quantization**. We do not just shrink the weights; we rebuild the LLaVA recipe around a tiny backbone and teach it to reason over images. By fusing the efficiency of **Qwen2.5-0.5B** with the sharp visual encoding of **SigLIP-Base**, we create a VLM that fits on a Raspberry Pi 5 or local laptop while retaining useful reasoning capabilities.

> **A note on "QLoRA"**: 4-bit NF4 training via `bitsandbytes` requires CUDA and has no Apple Silicon (MPS) backend. On the development machine (M2 Air), the student is therefore trained as **LoRA adapters on an FP16 frozen backbone**, and 4-bit is applied **post-training, at inference** (INT4 GGUF / MLX). The 4-bit QLoRA training path remains available for CUDA runs.

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
To democratize access to multimodal AI by proving that **architecture-aware compression** combined with **post-training 4-bit quantization** can yield edge-native VLMs that deliver a useful fraction of 7B-class visual reasoning while consuming **<2GB memory at inference** — and that the entire pipeline can be developed on an 8GB consumer laptop.

---

## Objectives

> Targets are stated as **measurable exit criteria**. Numbers marked *(target)* are not yet measured; the evaluation harness (Phase I, objective 4) is what turns them into results.

### Phase I — Architectural Compression & Efficient Transfer
1. **Rebuild the LLaVA architecture at ~1/12th the size**: SigLIP-Base (~93M) + Qwen2.5-0.5B-Instruct (~494M) + MLP projector (~1.5M) = **~0.59B total parameters**, of which ~10.3M are trainable (LoRA r=16 + projector).
2. **Correct, capacity-efficient SFT**: Train on LLaVA-Instruct-150K with answer-only loss over correctly interleaved multi-turn conversations — no gradient spent on padding, prompts, or scrambled turn order.
3. **Hardware-aware training pipeline**: An 8GB Apple Silicon (MPS) machine is the **development** target — pipeline validation, overfit gates, curated-subset training, and all inference/evaluation run locally. Full-dataset epochs run on a single commodity GPU (free-tier T4, ~3h/epoch); this split is a deliberate engineering decision, not a fallback.
4. **Evaluation before claims**: A held-out validation split, **POPE** (hallucination), and a fixed qualitative gallery — benchmarked against three reference points: the untrained student (floor), **SmolVLM-500M** (true peer), and LLaVA-v1.5-7B (ceiling).
5. **Edge compatibility**: The exported student runs in **<2GB memory** at FP16 (~1.2GB) and **<0.6GB** at INT4 (GGUF/MLX) — on Raspberry Pi 5, phones, and Apple Silicon.

**Explicitly out of scope for Phase I**: matching or "rivaling" LLaVA-v1.5-7B. A 0.59B student trained on 150K samples cannot, and claiming otherwise is not a research target but a marketing one. The honest target is **competitiveness with same-class models (SmolVLM-500M) and a measured, published gap to the 7B teacher.**

### Phase II — Hidden State Knowledge Distillation *(research phase, after Phase I ships)*
1. **Feature Alignment**: Implement Hidden State Distillation (HSD) mapping the Student's intermediate activations ($d=896$) to the Teacher's ($d=4096$) — bypassing the 32K/152K vocabulary mismatch by distilling *concepts* rather than *words*.
2. **Offline teacher traces**: Because the 7B teacher cannot co-reside with the student on an 8GB machine, teacher hidden states are precomputed **once** and compressed, making distillation a teacher-free local training run.
3. **Composite loss**: $L_{total} = (1 - \alpha) L_{CE} + \alpha L_{align}(H_{Teacher}, Proj(H_{Student}))$.
4. **Inference acceleration**: FlashAttention-2 (CUDA) / MLX-native attention (Apple Silicon) for low-latency edge serving.
---

## Methodology: 

The core innovation of Bonsai-LLaVA is the **"Pico" Architecture**, which serves as the foundation for both phases.

1. **The Common Architecture ("Pico")**
We replace the heavy legacy components of Standard LLaVA with highly efficient alternatives:

### Phase I: The "Pico" Architecture (Architecture Search)

We rebuilt the LLaVA architecture from scratch, swapping heavy legacy components for modern, lightweight alternatives designed for efficiency.


2. **Phase I Training Strategy: "Efficient SFT"**
In Phase I, we rely on **Data-Driven Transfer**:
**Precision**: The Qwen backbone is frozen in FP16 (CUDA runs may instead freeze it in 4-bit NF4 via QLoRA); trainable parameters are kept in FP32 for numerically stable optimizer updates.
**Optimization**: We train only the Projector and LoRA Adapters (~10.3M params, 1.7% of the model).
**Loss Function**: Cross-Entropy ($L_{CE}$) on **assistant answer tokens only** — prompts, chat-template scaffolding, and padding are masked out with `-100`.
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
Teacher Model: liuhaotian/llava-v1.5-7b (The Industry Standard)

Student Language Backbone: Qwen/Qwen2.5-0.5B-Instruct

Student Vision Backbone: google/siglip-base-patch16-224

### Optimization & Training
Framework: PyTorch 2.1+, Hugging Face Transformers.

Compression: LoRA adapters on a frozen FP16 backbone (MPS) / QLoRA 4-bit NF4 (CUDA); INT4 post-training quantization for inference.

Hardware: Apple Metal (MPS) for development, validation and inference; CUDA for full-dataset training runs.

### Inference Engine
Format: SafeTensors (FP16 or INT4)

Serving: vLLM (Planned)

---


## 📊 Targets (not yet measured)

| Metric | LLaVA-v1.5-7B | **Bonsai-LLaVA (Pico)** | Change |
| :--- | :--- | :--- | :--- |
| **Parameters** | 7.2 Billion | **~0.59 Billion** | 🔻 **92%** |
| **Trainable Params** | 7.2B (Full FT) | **~10.3M (1.7%)** | LoRA + Projector |
| **Memory (Training)** | ~80GB (Full FT) | **<8GB (LoRA, FP16)** | Consumer Ready |
| **Memory (Inference)** | ~14GB (FP16) | **~1.2GB (FP16) / ~0.5GB (INT4)** | **up to 28x Smaller** |
| **Architecture** | CLIP-L/336 + LLaMA-2 | SigLIP-B/224 + Qwen2.5 | Modern Stack |

> [!NOTE]
> Every row above is a **target**, not a result. Quality metrics (POPE, validation loss, qualitative gallery) are deliberately absent until the evaluation harness produces them — measured against SmolVLM-500M as the same-class peer. See `artifacts/bonsaillave-audit-report.md` for the full methodology and implementation audit driving the current work.
