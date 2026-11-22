"""Summarize local EML messages from a selected month with a FLAN-T5 model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email import message_from_bytes, policy
from email.message import Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

import torch

from src.email_common import MAX_INPUT_TOKENS, MAX_SUMMARY_TOKENS, tokenize_prompt
from src.model_utils import load_model_and_tokenizer, validate_device


@dataclass(frozen=True)
class EmailRecord:
    id: str
    date: datetime
    sender: str
    subject: str
    body: str
    source: str

    def prompt_data(self) -> dict[str, str]:
        return {"id": self.id, "from": self.sender, "subject": self.subject, "body": self.body}


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def message_body(message: Message) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in message.walk() if message.is_multipart() else (message,):
        if part.get_content_disposition() == "attachment":
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        content = part.get_content()
        if not isinstance(content, str):
            continue
        if part.get_content_type() == "text/plain":
            plain_parts.append(content)
        else:
            parser = HTMLTextExtractor()
            parser.feed(content)
            html_parts.append(" ".join(" ".join(parser.parts).split()))
    return "\n".join(plain_parts or html_parts).strip()


def load_month(input_path: Path, month: str) -> list[EmailRecord]:
    try:
        if len(month) != 7 or month[4] != "-" or not (month[:4] + month[5:]).isdigit():
            raise ValueError
        datetime.strptime(month, "%Y-%m")
    except ValueError as error:
        raise ValueError("month must be a valid calendar month in YYYY-MM format") from error

    if input_path.is_dir():
        files = sorted(path for path in input_path.rglob("*.eml") if path.is_file())
        if not files:
            raise ValueError(f"{input_path}: directory contains no .eml files")
    elif input_path.is_file() and input_path.suffix.lower() == ".eml":
        files = [input_path]
    else:
        raise ValueError(f"{input_path}: input must be an existing .eml file or directory of .eml files")

    selected = []
    for path in files:
        source = path.relative_to(input_path).as_posix() if input_path.is_dir() else path.name
        message = message_from_bytes(path.read_bytes(), policy=policy.default)
        raw_date = message.get("Date")
        if not raw_date:
            raise ValueError(f"{source}: email is missing a Date header")
        try:
            date = parsedate_to_datetime(str(raw_date))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{source}: email has an invalid Date header") from error
        if date is None:
            raise ValueError(f"{source}: email has an invalid Date header")
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if date.strftime("%Y-%m") != month:
            continue
        body = message_body(message)
        if not body:
            raise ValueError(f"{source}: email has no readable text body")
        sender = str(message.get("From", "")).strip() or "(unknown sender)"
        subject = str(message.get("Subject", "")).strip() or "(no subject)"
        selected.append((date, source, sender, subject, body))

    selected.sort(key=lambda item: (item[0], item[1]))
    return [
        EmailRecord(f"E{index:03d}", date, sender, subject, body, source)
        for index, (date, source, sender, subject, body) in enumerate(selected, start=1)
    ]


class EmailSummarizer(Protocol):
    def summarize(self, email: EmailRecord) -> tuple[str, bool]: ...


class FlanT5Summarizer:
    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def summarize(self, email: EmailRecord) -> tuple[str, bool]:
        encoded, truncated = tokenize_prompt(self.tokenizer, email.prompt_data())
        inputs = self.tokenizer.pad([encoded], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_SUMMARY_TOKENS)
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not summary:
            raise RuntimeError(f"model returned an empty summary for [{email.id}]")
        return summary, truncated


def summarize_emails(emails: list[EmailRecord], summarizer: EmailSummarizer) -> list[tuple[str, bool]]:
    return [summarizer.summarize(email) for email in emails]


def render_report(month: str, emails: list[EmailRecord], summaries: list[tuple[str, bool]]) -> str:
    if len(emails) != len(summaries):
        raise ValueError("each email must have exactly one summary")
    lines = [
        f"# Email summaries: {month}",
        "",
        "Generated summaries may be inaccurate. Verify them against the source emails.",
    ]
    for email, (summary, truncated) in zip(emails, summaries):
        lines.extend([
            "",
            f"## [{email.id}] {email.subject}",
            "",
            f"- Date: {email.date.isoformat()}",
            f"- From: {email.sender}",
            f"- Source: {email.source}",
        ])
        if truncated:
            lines.append(f"- Warning: input exceeded {MAX_INPUT_TOKENS} tokens; the end was omitted.")
        lines.extend(["", summary])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help=".eml file or directory of .eml files")
    parser.add_argument("--month", required=True, help="month to summarize, in YYYY-MM format")
    parser.add_argument("--model", required=True, help="saved model directory or Hugging Face model ID")
    parser.add_argument("--output", required=True, type=Path, help="new Markdown report path")
    parser.add_argument("--local-files-only", action="store_true", help="do not download model files")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not already exist")
    emails = load_month(args.input, args.month)
    if not emails:
        parser.error(f"no emails found for {args.month}")
    try:
        validate_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.local_files_only)
    summarizer = FlanT5Summarizer(model, tokenizer, args.device)
    summaries = summarize_emails(emails, summarizer)
    report = render_report(args.month, emails, summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(report)
    print(f"Wrote {len(emails)} email summary/summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
