# FLAN-T5 email summarization

A small prototype for fine-tuning Google's FLAN-T5 on labeled emails and
generating summaries of local email exports. The inference command selects one
month, summarizes each email separately, and groups the results in a Markdown
report. It does **not** train or generate a single conversation-level monthly
summary.

The five fictional examples in this repository demonstrate the workflow; they
are far too few to establish model quality. For a useful domain-specific model,
prepare a larger, reviewed dataset and evaluate it on separate email threads or
months.

## Project files

| Path | Purpose |
| --- | --- |
| `src/train_email_summarizer.py` | Fine-tune and save a sequence-to-sequence model and tokenizer |
| `src/email_digest.py` | Summarize local `.eml` files with a saved or pretrained model |
| `src/email_common.py` | Shared prompt and input tokenization |
| `src/model_utils.py` | Shared model loading and device validation |
| `examples/sample_email_training.jsonl` | Five fictional labeled training examples |
| `examples/sample_emails/` | Five fictional `.eml` messages for January 2025 |
| `tests/` | Unit and small local-model integration tests |
| `requirements.txt` | Python dependencies |

## Setup

Use Python 3.10 or later and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
PyTorch, Transformers and SentencePiece are required. The commands default to
CPU; `--device cuda` and `--device mps` are available when supported by the
installed PyTorch build and hardware. Training can be slow on CPU and requires
enough memory for the model.

By default, training loads `google/flan-t5-base` from Hugging Face. The model
and tokenizer are not included in this repository. Initial downloads need
network access and may be subject to Hugging Face availability and model terms.
If access is restricted, use a complete compatible model and tokenizer from an
approved local source or cache, pass its path with `--model`, and add
`--local-files-only`. This prevents Hugging Face downloads; dependency
installation is a separate step.

## Training data

Training requires UTF-8 JSON Lines: one JSON object per line, with nonempty
string fields `id`, `from`, `subject`, `body`, and `summary`. IDs must be `E`
followed by digits, and each summary must contain its ID in square brackets:

```json
{"id":"E001","from":"Alex <alex@example.test>","subject":"Project timeline","body":"Please review the timeline by Friday.","summary":"Alex requests a timeline review by Friday. [E001]"}
```

`examples/sample_email_training.jsonl` contains five labeled examples
corresponding to the fictional sample emails. The `.eml` files alone are
inference inputs, not labeled training data. For real fine-tuning, write
accurate human-reviewed summaries of emails you are authorized to use. Create
a separate held-out JSONL file in the same format, with separate threads or
months to reduce leakage. Do not treat the five sample examples as a meaningful
evaluation set.

The shared prompt asks the model for up to two sentences about decisions,
requests, deadlines, and follow-ups, with a source ID. The input is limited to
512 tokens and the target summary to 96 tokens. Training prints warnings when
either is truncated. Long input is cut at the end, potentially losing important
context; shorten or prepare appropriate examples before training. IDs in
generated text are not guaranteed to be correct.

## Fine-tune

For a small demonstration using the fictional data:

```bash
python -m src.train_email_summarizer \
  --train-file examples/sample_email_training.jsonl \
  --output-dir ./models/email-flan-t5 \
  --device cpu
```

The output directory must be new or empty. The script saves the trained model
and tokenizer there. Default settings are two epochs, batch size 2, learning
rate `5e-5`, and seed 42. For your own data, an optional held-out file reports
evaluation loss after each epoch:

```bash
python -m src.train_email_summarizer \
  --train-file /path/to/private/train.jsonl \
  --eval-file /path/to/private/held-out.jsonl \
  --output-dir /path/to/private/email-model \
  --model /path/to/local/flan-t5-base \
  --local-files-only \
  --epochs 2 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --device cpu
```

The evaluation file must be a different file from the training file. Loss
alone does not measure factual accuracy, relevance, coverage or usefulness.
Manually compare generated summaries with held-out source emails. See
`python -m src.train_email_summarizer --help` for all options.

## Run inference

After fine-tuning, summarize the fictional January emails:

```bash
python -m src.email_digest \
  --input examples/sample_emails \
  --month 2025-01 \
  --model ./models/email-flan-t5 \
  --local-files-only \
  --output ./outputs/january.md
```

Alternatively, pass a single email:

```bash
python -m src.email_digest \
  --input examples/sample_emails/01-project-timeline.eml \
  --month 2025-01 \
  --model ./models/email-flan-t5 \
  --local-files-only \
  --output ./outputs/one-email.md
```

`--model` can also be a compatible pretrained Hugging Face model ID, which
requires access unless already cached. Specify `--device cuda` or `--device
mps` if appropriate. The report path must not already exist. A directory is
searched recursively for lowercase `*.eml` files; `.mbox` files and direct
email-account connections are not supported. Each message needs a valid `Date`
header and a readable text body; attachments are not summarized. Plain text is
preferred over HTML. The month is based on the message's stated timezone;
timestamps without a timezone are treated as UTC.

The report includes the source file, sender, timestamp, subject, and generated
summary for each selected email. Each input is limited to 512 tokens, and the
report flags emails whose ends were omitted. The model may invent facts, miss
important details, or generate incorrect source IDs. Report headings identify
the actual source independently of the generated text; review every result
against the original. This prototype does not group messages into threads or
guarantee protection against instructions embedded in emails.

## Privacy and evaluation

Use fictional examples until you have permission to process real messages.
Email exports, labels, generated reports and model weights may contain or
memorize sensitive information. Keep them on trusted storage outside the
repository and do not commit them. The ignore rules cover only common local
paths and do not protect arbitrary files. Inference processes local files and
does not connect to an email account; model downloads, if enabled, contact
Hugging Face.

To run the tests in an environment with the dependencies installed:

```bash
python -m unittest discover -s tests
```

The integration test constructs a tiny local model; it does not download the
FLAN-T5 base model. The tests do not establish summarization quality.
