import math
import torch
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import os 

class Config:
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    dataset_dict = {"TEXT_ONLY": "", "PAIRED": "", "IMAGE_ONLY": ""}

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_dtype(name: str):
    name = (name or "").lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    return torch.bfloat16


def get_model_and_processor(
    config: Config,
    model_id: str,
    quantization: dict | None = None,
    gradient_checkpointing: bool = False,
):
    processor = AutoProcessor.from_pretrained(
        model_id, use_fast=True, min_pixels=config.min_pixels, max_pixels=config.max_pixels
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quant_cfg_obj = None
    if quantization is not None:
        q = dict(quantization)
        if isinstance(q.get("bnb_4bit_compute_dtype"), str):
            q["bnb_4bit_compute_dtype"] = get_dtype(q["bnb_4bit_compute_dtype"])
        quant_cfg_obj = BitsAndBytesConfig(**q)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg_obj,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    if gradient_checkpointing:
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, processor
