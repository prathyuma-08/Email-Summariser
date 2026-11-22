"""Fine-tune a Hugging Face sequence-to-sequence model on email summaries."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from src.email_common import MAX_INPUT_TOKENS, MAX_SUMMARY_TOKENS, tokenize_prompt
from src.model_utils import load_model_and_tokenizer, validate_device


FIELDS = ("id", "from", "subject", "body", "summary")


def load_examples(path: Path) -> list[dict[str, str]]:
    examples = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(example, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            for field in FIELDS:
                if not isinstance(example.get(field), str) or not example[field].strip():
                    raise ValueError(f"{path}:{line_number}: {field} must be a nonempty string")
            if not re.fullmatch(r"E\d+", example["id"]):
                raise ValueError(f"{path}:{line_number}: id must look like E001")
            if f"[{example['id']}]" not in example["summary"]:
                raise ValueError(f"{path}:{line_number}: summary must cite [{example['id']}]")
            examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no training examples found")
    return examples


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path, help="training examples in JSONL format")
    parser.add_argument("--eval-file", type=Path, help="optional held-out JSONL examples")
    parser.add_argument("--output-dir", required=True, type=Path, help="new or empty model output directory")
    parser.add_argument("--model", default="google/flan-t5-base", help="Hugging Face model ID or local path")
    parser.add_argument("--local-files-only", action="store_true", help="do not download model files")
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--learning-rate", type=positive_float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("--output-dir must be a new or empty directory")
    if args.eval_file and args.eval_file.resolve() == args.train_file.resolve():
        parser.error("--eval-file must be separate from --train-file")
    try:
        validate_device(args.device)
    except ValueError as error:
        parser.error(str(error))


def tokenize_examples(examples: list[dict[str, str]], tokenizer) -> list[dict]:
    dataset = []
    for example in examples:
        inputs, truncated = tokenize_prompt(tokenizer, example)
        if truncated:
            print(
                f"Warning: input for [{example['id']}] exceeds {MAX_INPUT_TOKENS} tokens and was truncated",
                file=sys.stderr,
            )
        labels = tokenizer(text_target=example["summary"], truncation=False)
        if len(labels["input_ids"]) > MAX_SUMMARY_TOKENS:
            print(
                f"Warning: summary for [{example['id']}] exceeds {MAX_SUMMARY_TOKENS} tokens and was truncated",
                file=sys.stderr,
            )
            labels = tokenizer(
                text_target=example["summary"], truncation=True, max_length=MAX_SUMMARY_TOKENS
            )
        inputs["labels"] = labels["input_ids"]
        dataset.append(inputs)
    return dataset


def create_data_loaders(train_examples, eval_examples, tokenizer, model, batch_size: int, seed: int):
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_data = tokenize_examples(train_examples, tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collator
    )
    eval_loader = (
        DataLoader(tokenize_examples(eval_examples, tokenizer), batch_size=batch_size, collate_fn=collator)
        if eval_examples else None
    )
    return train_loader, eval_loader


def run_epoch(model, loader, device: str, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad()
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                phase = "Training" if training else "Evaluation"
                raise RuntimeError(f"{phase} loss is not finite; model was not saved")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["input_ids"].shape[0]
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, eval_loader, *, device: str, epochs: int, learning_rate: float) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        print(f"Epoch {epoch + 1}/{epochs}: training loss {train_loss:.4f}")
        if eval_loader:
            eval_loss = run_epoch(model, eval_loader, device)
            print(f"  held-out loss {eval_loss:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    train_examples = load_examples(args.train_file)
    eval_examples = load_examples(args.eval_file) if args.eval_file else None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.local_files_only)
    train_loader, eval_loader = create_data_loaders(
        train_examples, eval_examples, tokenizer, model, args.batch_size, args.seed
    )
    train_model(
        model, train_loader, eval_loader, device=args.device,
        epochs=args.epochs, learning_rate=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model and tokenizer to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
