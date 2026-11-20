# Beyond-Accuracy

Dataset link: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

## Monthly email digest prototype

The optional `src/email_digest.py` command creates a digest from local `.eml` files,
an `.mbox` archive, or a directory containing `.eml` files. It does not connect to
an email account. It sends email text to an Ollama server on `127.0.0.1:11434`
for inference with a locally installed model. This prototype does not train a model.

Start Ollama and download the default model once:

```bash
ollama pull qwen2.5:3b
ollama list
```

The model download requires network access to the Ollama registry. If it is
blocked, make a locally downloaded model available to Ollama by an approved
method and pass its name with `--model`; `ollama list` shows available models.
This command uses only Python's standard library and does not require the
Hugging Face dependencies. Choose a trusted local model, not an Ollama cloud
model: inference privacy depends on your Ollama configuration and model choice.
Model files are not included in this repository.

Create a report for one month:

```bash
python src/email_digest.py \
  --input /path/to/email-export.mbox \
  --month 2025-01 \
  --output /path/to/january-digest.md
```

For individual `.eml` files, pass a directory containing them as `--input`.
The report includes the generated digest and an index of source emails. Generated
content can be inaccurate; verify it against the referenced messages. Email
content is treated as untrusted input and may contain private information, so
use a trusted local machine and do not commit exports or reports.

To try the tool without using personal email, the repository includes fictional
messages in `examples/sample_emails/`. After installing a model in Ollama, run:

```bash
python src/email_digest.py \
  --input examples/sample_emails \
  --month 2025-01 \
  --output ~/sample-email-digest.md
```

These are sample input messages, not labeled training data. Fine-tuning requires
email-and-summary pairs that you have reviewed and a separate training framework;
Ollama serves models but does not train model weights. Loading a tokenizer, model,
and data collator alone is not training.

## Fine-tuning with Hugging Face

`src/train_email_summarizer.py` fine-tunes a sequence-to-sequence model such as
`google/flan-t5-base` using PyTorch and Transformers. Install the optional
training dependencies:

```bash
python -m pip install -r requirements-email-digest.txt
```

The training file is JSON Lines: each line contains `id`, `from`, `subject`,
`body`, and `summary` strings; the summary must cite the source ID. See
`examples/sample_email_training.jsonl` for five **fictional** examples. They
are only for demonstrating the format, not a sufficient training set. Prepare
your own reviewed examples, and keep personal training data and model outputs
outside the repository; model weights can memorize private information.

```bash
python src/train_email_summarizer.py \
  --train-file ~/email-slm-private/train.jsonl \
  --eval-file ~/email-slm-private/held-out.jsonl \
  --output-dir ~/email-slm-private/model \
  --device cpu
```

The optional `--eval-file` reports held-out loss; use examples from separate
threads or months and manually compare generated summaries for factual accuracy
and citations. Use `--device mps` on a supported Apple Silicon Mac, or omit
`--eval-file` if you have not prepared held-out data yet. Output must be a new
or empty directory. Run `python src/train_email_summarizer.py --help` for options.

The default model must first be available from Hugging Face or in the local
cache. If downloads are blocked, obtain the complete pretrained model and
tokenizer files from a trusted source by an approved method and pass their
directory using `--model /path/to/model --local-files-only`. Training and
evaluation use local files; downloading the base model requires network access.

The saved model is a Hugging Face FLAN-T5 model. It does **not** automatically
replace the Ollama model used by the digest command; the two use different model
formats and this project does not provide a conversion or inference path for
the fine-tuned FLAN-T5 model.

Run the unit tests with:

```bash
python -m unittest discover -s tests
```