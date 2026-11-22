from contextlib import nullcontext
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.email_common import MAX_INPUT_TOKENS, format_prompt, tokenize_prompt
from src.email_digest import (
    EmailRecord,
    FlanT5Summarizer,
    load_month,
    main,
    render_report,
    summarize_emails,
)


def make_message(date: str, subject: str, body: str) -> bytes:
    message = EmailMessage()
    message["Date"] = date
    message["From"] = "sender@example.test"
    message["Subject"] = subject
    message.set_content(body)
    return message.as_bytes()


class FakeTensor:
    def to(self, device: str):
        return self


class FakeTokenizer:
    def __init__(self, token_count: int = 20, decoded: str = "A summary [E001].") -> None:
        self.token_count = token_count
        self.decoded = decoded
        self.prompts = []

    def __call__(self, text: str, truncation: bool, max_length: int | None = None):
        self.prompts.append(text)
        if truncation:
            assert max_length is not None
            count = min(self.token_count, max_length)
        else:
            count = self.token_count
        return {"input_ids": list(range(count))}

    def pad(self, encoded, return_tensors):
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    def decode(self, tokens, skip_special_tokens):
        return self.decoded


class FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[1, 2]]


class EmailDigestTests(unittest.TestCase):
    def test_sample_emails_are_readable_for_documented_month(self) -> None:
        sample_dir = Path(__file__).resolve().parents[1] / "examples" / "sample_emails"
        emails = load_month(sample_dir, "2025-01")
        self.assertEqual(len(emails), 5)
        self.assertEqual(emails[0].subject, "Project timeline review")
        self.assertEqual(emails[-1].subject, "January community update")
        self.assertEqual(load_month(sample_dir, "2025-02"), [])

    def test_load_month_filters_and_orders_eml_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "second.eml").write_bytes(
                make_message("Mon, 20 Jan 2025 09:00:00 +0000", "Later", "Second message")
            )
            first = root / "first.eml"
            first.write_bytes(make_message("Fri, 03 Jan 2025 09:00:00 +0000", "Earlier", "First message"))
            (root / "other.eml").write_bytes(
                make_message("Sat, 01 Feb 2025 09:00:00 +0000", "Other month", "Ignore")
            )
            emails = load_month(root, "2025-01")
            self.assertEqual([email.subject for email in emails], ["Earlier", "Later"])
            self.assertEqual([email.id for email in emails], ["E001", "E002"])
            self.assertEqual(load_month(first, "2025-01")[0].source, "first.eml")

    def test_invalid_input_and_month_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "valid calendar month"):
                load_month(root, "2025-13")
            with self.assertRaisesRegex(ValueError, "no .eml files"):
                load_month(root, "2025-01")
            with self.assertRaisesRegex(ValueError, "existing .eml file"):
                load_month(root / "missing.eml", "2025-01")
            (root / "invalid.eml").write_bytes(b"Subject: no date\n\nbody")
            with self.assertRaisesRegex(ValueError, "missing a Date"):
                load_month(root, "2025-01")

    def test_summary_uses_shared_prompt_and_reports_truncation(self) -> None:
        email = EmailRecord("E001", datetime(2025, 1, 3, tzinfo=timezone.utc),
                            "sender@example.test", "Timeline", "Please confirm.", "first.eml")
        tokenizer = FakeTokenizer(MAX_INPUT_TOKENS + 1)
        with patch("src.email_digest.torch.inference_mode", return_value=nullcontext()):
            summary, truncated = FlanT5Summarizer(FakeModel(), tokenizer, "cpu").summarize(email)
        self.assertEqual(tokenizer.prompts[0], format_prompt(email.prompt_data()))
        self.assertTrue(truncated)
        self.assertEqual(summary, "A summary [E001].")
        report = render_report("2025-01", [email], [(summary, truncated)])
        self.assertIn("first.eml", report)
        self.assertIn("the end was omitted", report)
        self.assertIn(summary, report)

    def test_exact_input_limit_does_not_truncate(self) -> None:
        tokenizer = FakeTokenizer(MAX_INPUT_TOKENS)
        example = {"id": "E001", "from": "a", "subject": "b", "body": "c"}
        encoded, truncated = tokenize_prompt(tokenizer, example)
        self.assertFalse(truncated)
        self.assertEqual(len(encoded["input_ids"]), MAX_INPUT_TOKENS)
        self.assertEqual(len(tokenizer.prompts), 1)

    def test_empty_model_output_and_overwrite_are_rejected(self) -> None:
        email = EmailRecord("E001", datetime(2025, 1, 3, tzinfo=timezone.utc),
                            "sender@example.test", "Timeline", "body", "first.eml")
        with patch("src.email_digest.torch.inference_mode", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "empty summary"):
                FlanT5Summarizer(FakeModel(), FakeTokenizer(decoded=" "), "cpu").summarize(email)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(["--input", "unused.eml", "--month", "2025-01",
                      "--model", "unused", "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_inference_cli_loads_model_and_writes_report(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        sample = Path(__file__).resolve().parents[1] / "examples" / "sample_emails" / "01-project-timeline.eml"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            with patch("src.email_digest.load_model_and_tokenizer", return_value=(model, tokenizer)) as load:
                with patch("src.email_digest.torch.inference_mode", return_value=nullcontext()):
                    result = main([
                        "--input", str(sample), "--month", "2025-01",
                        "--model", "/local/model", "--local-files-only", "--output", str(output),
                    ])
            self.assertEqual(result, 0)
            self.assertIn("Project timeline review", output.read_text(encoding="utf-8"))
        load.assert_called_once_with("/local/model", "cpu", True)
        model.eval.assert_called_once()

    def test_summary_orchestration_accepts_an_alternative_summarizer(self) -> None:
        email = EmailRecord("E001", datetime(2025, 1, 3, tzinfo=timezone.utc),
                            "sender@example.test", "Timeline", "body", "first.eml")
        summarizer = Mock()
        summarizer.summarize.return_value = ("Alternative summary", False)
        self.assertEqual(summarize_emails([email], summarizer), [("Alternative summary", False)])
        summarizer.summarize.assert_called_once_with(email)


if __name__ == "__main__":
    unittest.main()
