import unittest
from unittest.mock import Mock, patch

from src.model_utils import load_model_and_tokenizer, validate_device


class ModelUtilsTests(unittest.TestCase):
    @patch("src.model_utils.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("src.model_utils.AutoTokenizer.from_pretrained")
    def test_loads_model_and_tokenizer_with_same_source(self, load_tokenizer, load_model) -> None:
        model = Mock()
        model.to.return_value = model
        load_model.return_value = model
        tokenizer = load_tokenizer.return_value
        self.assertEqual(load_model_and_tokenizer("/model", "cpu", True), (model, tokenizer))
        load_tokenizer.assert_called_once_with("/model", local_files_only=True)
        load_model.assert_called_once_with("/model", local_files_only=True)
        model.to.assert_called_once_with("cpu")

    @patch("src.model_utils.torch.cuda.is_available", return_value=False)
    def test_rejects_unavailable_cuda(self, is_available) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            validate_device("cuda")
        is_available.assert_called_once()


if __name__ == "__main__":
    unittest.main()
