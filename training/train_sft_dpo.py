"""
Stage 3.1: Open-Weight QLoRA Supervised Fine-Tuning (SFT) + Direct Preference Optimization (DPO)
Target Architecture: Qwen/Qwen3-14B (with Qwen/Qwen3-4B fast-tier support)
Hardware Target: Kaggle Dual NVIDIA Tesla T4 GPUs (2x 16GB = 32GB GDDR6, sm_75)
Author: Vignesh (VigneshCEO / SRIM Works)
"""
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "checkpoints"


def format_chatml(example, tokenizer):
    messages = example.get("messages", [])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def run_sft_and_dpo_pipeline(
    base_model_id: str = "Qwen/Qwen3-14B",
    sft_file: str = None,
    dpo_file: str = None,
    output_dir: str = None,
    max_seq_length: int = 2048,
    sft_epochs: int = 3,
    dpo_epochs: int = 2,
):
    sft_path = str(sft_file or (DATA_DIR / "ven_sft_train.jsonl"))
    dpo_path = str(dpo_file or (DATA_DIR / "ven_dpo_pairs.jsonl"))
    out_root = Path(output_dir or OUTPUT_DIR)
    sft_out = out_root / "ven_hermes_sft_lora"
    dpo_out = out_root / "ven_hermes_dpo_lora"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 3.1] Loading tokenizer and 4-bit NF4 base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # -------------------------------------------------------------------------
    # Phase A: Supervised Fine-Tuning (QLoRA SFT) across all 5 VEN Tasks
    # -------------------------------------------------------------------------
    print(f"[Stage 3.1A] Loading SFT dataset from {sft_path}...")
    sft_ds = load_dataset("json", data_files=sft_path, split="train")
    sft_ds = sft_ds.map(lambda ex: format_chatml(ex, tokenizer))

    sft_args = SFTConfig(
        output_dir=str(sft_out),
        num_train_epochs=sft_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    t0 = time.time()
    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_args,
    )
    sft_res = sft_trainer.train()
    sft_trainer.model.save_pretrained(str(sft_out))
    tokenizer.save_pretrained(str(sft_out))
    sft_elapsed = round(time.time() - t0, 2)
    print(f"[Stage 3.1A] Completed QLoRA SFT in {sft_elapsed}s | Loss={sft_res.training_loss:.4f}")

    # -------------------------------------------------------------------------
    # Phase B: Direct Preference Optimization (DPO) Alignment
    # -------------------------------------------------------------------------
    print(f"[Stage 3.1B] Loading DPO preference pairs from {dpo_path}...")
    dpo_ds = load_dataset("json", data_files=dpo_path, split="train")

    dpo_args = DPOConfig(
        output_dir=str(dpo_out),
        num_train_epochs=dpo_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_8bit",
        max_length=max_seq_length,
        max_prompt_length=1024,
        report_to="none",
    )

    t1 = time.time()
    dpo_trainer = DPOTrainer(
        model=sft_trainer.model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )
    dpo_res = dpo_trainer.train()
    dpo_trainer.model.save_pretrained(str(dpo_out))
    tokenizer.save_pretrained(str(dpo_out))
    dpo_elapsed = round(time.time() - t1, 2)
    print(f"[Stage 3.1B] Completed DPO Alignment in {dpo_elapsed}s | Loss={dpo_res.training_loss:.4f}")

    metrics = {
        "base_model": base_model_id,
        "lora_rank": 32,
        "lora_alpha": 64,
        "sft_examples": len(sft_ds),
        "dpo_pairs": len(dpo_ds),
        "sft_training_loss": round(float(sft_res.training_loss), 4),
        "dpo_training_loss": round(float(dpo_res.training_loss), 4),
        "sft_elapsed_sec": sft_elapsed,
        "dpo_elapsed_sec": dpo_elapsed,
    }
    (out_root / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    run_sft_and_dpo_pipeline()
