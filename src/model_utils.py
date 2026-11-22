"""Shared model and device setup for training and inference."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def validate_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; choose --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; choose --device cpu")


def load_model_and_tokenizer(model_name: str, device: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device)
    return model, tokenizer
