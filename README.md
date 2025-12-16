# Bonsai-LLaVA: Distilling & Quantizing Multimodal Giants

> ** Bonsai-Llava** project is started with a sole purpose to practice the **Art of miniaturization without losing the Essence**.

> It's a research practice to distill the visual reasoning capabilities of **LLaVA-v1.5-7B** into a compact **<1B param(490M) student** (TinyLlama + SigLIP) model, optimized for consumer hardware and edge inference text-to-image generation.

## Introduction
The current state of Vision-Language Models (VLMs) is dominated by massive parameters (7B+), making them inaccessible for real-time edge applications or resource-constrained environments. While standard quantization helps, it often degrades semantic reasoning.

**Bonsai-LLaVA** solves this by combining **Knowledge Distillation** with **QLoRA**. We do not just shrink the weights; we teach a tiny, 4-bit quantized "Student" model to mimic the reasoning process of a "Teacher" giant. By fusing the efficiency of **Qwen2.5-0.5B** with the sharp visual encoding of **SigLIP-Base**, we create a VLM that fits on a Raspberry Pi 5 or local laptop while retaining strong reasoning capabilities.

## 🚀 Mission
To democratize access to high-performance multimodal AI by proving that **architecture-aware distillation** combined with **4-bit Quantization (QLoRA)** can yield edge-native models that rival 7B counterparts while consuming **<2GB VRAM**.

---

## 🎯 Objectives
1.  **Construct "Pico" Architecture:** Fuse `Qwen2.5-0.5B-Instruct` (Language) with `SigLIP-Base-Patch16-224` (Vision) using a custom MLP projector, keeping total parameters under **0.7B**.
2.  **Implement QLoRA Distillation:** Train the student using **Quantized Low-Rank Adaptation**. The student backbone is frozen in 4-bit NF4 precision, and only the adapters are trained to minimize the **KL-Divergence** between the Student's and Teacher's logits.
3.  **Optimize for Inference:** Native integration with **FlashAttention-2** and **vLLM** for production serving, targeting <20ms latency.

---
