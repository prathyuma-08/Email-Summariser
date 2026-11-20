import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.train_email_summarizer import format_prompt, load_examples, main


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "examples" / "sample_email_training.jsonl"


class TrainingDataTests(unittest.TestCase):
    def test_sample_data_has_valid_labels_and_matching_prompt(self) -> None:
        examples = load_examples(SAMPLE_DATA)
        self.assertEqual(len(examples), 5)
        self.assertIn("Source ID: [E001]", format_prompt(examples[0]))
        self.assertIn(examples[0]["body"], format_prompt(examples[0]))

    def test_rejects_bad_json_and_missing_citation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text("{invalid}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_examples(path)
            path.write_text(json.dumps({
                "id": "E001", "from": "a", "subject": "b",
                "body": "c", "summary": "Missing citation",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary must cite"):
                load_examples(path)

    def test_refuses_to_overwrite_a_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                main(["--train-file", str(SAMPLE_DATA), "--output-dir", str(ROOT)])


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers")),
    "Hugging Face training dependencies are not installed",
)
class TrainingIntegrationTests(unittest.TestCase):
    def test_fine_tunes_and_saves_tiny_local_model(self) -> None:
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            output = root / "output"
            evaluation = root / "eval.jsonl"
            evaluation.write_text(json.dumps({
                "id": "E006", "from": "a", "subject": "b",
                "body": "email", "summary": "summary [E006]",
            }) + "\n", encoding="utf-8")
            base.mkdir()
            tokenizer_backend = Tokenizer(WordLevel(
                {"<pad>": 0, "</s>": 1, "<unk>": 2, "email": 3, "summary": 4},
                unk_token="<unk>",
            ))
            tokenizer_backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer_backend,
                pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
            )
            tokenizer.save_pretrained(base)
            model = T5ForConditionalGeneration(T5Config(
                vocab_size=5, d_model=16, d_ff=32, num_layers=1,
                num_decoder_layers=1, num_heads=2,
                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
            ))
            before = model.shared.weight.detach().clone()
            model.save_pretrained(base)

            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                result = main([
                    "--train-file", str(SAMPLE_DATA),
                    "--eval-file", str(evaluation),
                    "--model", str(base), "--local-files-only",
                    "--output-dir", str(output), "--epochs", "1",
                ])
            finally:
                torch.set_num_threads(old_threads)

            self.assertEqual(result, 0)
            saved = T5ForConditionalGeneration.from_pretrained(output, local_files_only=True)
            self.assertFalse(torch.equal(before, saved.shared.weight.detach()))
            self.assertTrue((output / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
