"""Fine-tune T5-small for OS-command semantic parsing.

**Reconstruction.** The original Colab notebook and the 48k-example synthetic
dataset were lost. This script reproduces the training setup from the
hyper-parameters stored with the released weights (``training_args.json``,
extracted from ``training_args.bin``) and from the inference code (prefix
``semantic_parse: ``, flat ``"key":"value"`` targets sorted by key). It has not
been re-run end to end; it is provided so the model can be re-trained on a new
dataset with the same format.

Data format (JSON lines)::

    {"input": "open firefox and go to github.com",
     "target": {"intent": "open_app_and_go_url", "app": "firefox", "url": "github.com"}}

Usage::

    python training/train.py --train-file data/train.jsonl --output-dir models/my-t5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIX = "semantic_parse: "
HPARAMS = json.loads((Path(__file__).with_name("training_args.json")).read_text())


def to_target_text(target: dict) -> str:
    """Canonical target string, identical to the model's output format."""
    return ",".join(f'"{k}":"{target[k]}"' for k in sorted(target))


def main() -> None:
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--base-model", default="t5-small")
    ap.add_argument("--output-dir", default="models/t5-semantic-parser")
    ap.add_argument("--max-input-len", type=int, default=64)
    ap.add_argument("--max-target-len", type=int, default=128)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    ds = load_dataset("json", data_files={"train": args.train_file})["train"]

    def preprocess(batch):
        enc = tokenizer([PREFIX + x for x in batch["input"]], max_length=args.max_input_len, truncation=True)
        labels = tokenizer(
            text_target=[to_target_text(t) for t in batch["target"]], max_length=args.max_target_len, truncation=True
        )
        enc["labels"] = labels["input_ids"]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        **{k: v for k, v in HPARAMS.items() if not k.startswith("_")},
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
