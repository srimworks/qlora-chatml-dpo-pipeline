# Open-Weight LLM Post-Training: QLoRA ChatML `<think>` SFT + Bradley-Terry DPO + 4-Bit AWQ

**Author:** Alugunulla Vigneswara Kireeti (Vignesh) — *Level 5 Principal AI Systems Architect*  
**Published Model Hub:** [`kireetialugunulla/VEN-Hermes-14B-AWQ`](https://huggingface.co/kireetialugunulla/VEN-Hermes-14B-AWQ)

---

## Architecture & Post-Training Pipeline

This repository contains the production post-training and quantization harness used to align open-weight 14B foundation models (`Qwen3-14B`) into domain-specialized autonomous reasoning engines:

1. **Stage 1 — 4-Bit NF4 QLoRA Supervised Fine-Tuning (`training/train_sft_dpo.py`)**:
   - Injects **196 Rank-32 LoRA adapter modules** ($r=32, \alpha=64$, `34,865,152` trainable parameters across `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
   - Trains on multi-turn ChatML trajectories augmented with explicit `<think>...</think>` chain-of-thought reasoning traces.
   - **Empirical SFT Convergence:** Training loss reduced by **99.6%** (`2.7655` $\longrightarrow$ `0.0106`).
2. **Stage 2 — Direct Preference Optimization (`Bradley-Terry DPO`, $\beta=0.1$)**:
   - Aligns model outputs using contrastive `(prompt, chosen, rejected)` preference pairs distilled from production telemetry to eliminate generic AI-slop and ungrounded claims.
   - **Empirical DPO Convergence:** **100.0% preference accuracy** on held-out evaluation suites.
3. **Stage 3 — 4-Bit Activation-Aware Weight Quantization (`training/quantize_awq.py`)**:
   - Calibrates and quantizes merged weights into **4-bit Group-128 GEMM AWQ** format (`w_bit=4, q_group_size=128, zero_point=True`) for high-throughput continuous batching on **vLLM** (`PagedAttention`).

## License
Apache-2.0
