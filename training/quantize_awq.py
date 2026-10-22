"""
Stage 3.2: 4-Bit AWQ / NF4 Quantization & Hugging Face Hub Deployment
Merges QLoRA + DPO adapters, performs 4-bit Activation-Aware Weight Quantization (AWQ: w_bit=4, q_group_size=128),
and publishes model artifacts, custom C++/CUDA kernels (`ven_cuda_kernels.cu`), and hardware telemetry to:
  https://huggingface.co/VigneshCEO/VEN-Hermes-14B-AWQ
Author: Vignesh (VigneshCEO / SRIM Works)
"""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).resolve().parent.parent
CSRC_DIR = BASE_DIR / "csrc"
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

HF_REPO_ID = "kireetialugunulla/VEN-Hermes-14B-AWQ"
HF_MIRROR_REPO_ID = "srimworksa/VEN-Hermes-14B-AWQ"


def quantize_to_4bit_awq(merged_model_dir: str, output_awq_dir: str, calib_jsonl: str):
    """
    Calibrates and quantizes the merged QLoRA+DPO checkpoint into 4-bit AWQ GEMM format
    (w_bit=4, q_group_size=128, zero_point=True) using live VEN calibration prompts.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    calib_samples = []
    with open(calib_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages", [])
            if msgs:
                calib_samples.append("\n".join(m.get("content", "") for m in msgs))
            if len(calib_samples) >= 128:
                break

    model = AutoAWQForCausalLM.from_pretrained(merged_model_dir, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_samples)
    model.save_quantized(output_awq_dir)
    tokenizer.save_pretrained(output_awq_dir)
    return quant_config


def publish_to_huggingface_hub(
    hf_token: str,
    repo_id: str = HF_REPO_ID,
    artifacts_dir: Path = None,
    cuda_benchmark: dict = None,
    training_metrics: dict = None,
):
    create_repo(repo_id=repo_id, token=hf_token, repo_type="model", exist_ok=True, private=False)
    api = HfApi(token=hf_token)

    if artifacts_dir and artifacts_dir.exists():
        api.upload_folder(
            folder_path=str(artifacts_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload VEN-Hermes-14B-AWQ Level 5 QLoRA+DPO weights, C++/CUDA kernels & telemetry",
        )
    print(f"[Stage 3.2] Published artifacts to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"
