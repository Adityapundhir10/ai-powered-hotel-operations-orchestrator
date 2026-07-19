"""Fine-tune multilingual DistilBERT for joint intent/severity labels.

Expected CSV columns: text,label, where label uses intent__severity, for example
maintenance__high. This script is intentionally not run during installation.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/distilbert-checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    frame = pd.read_csv(args.data).dropna(subset=["text", "label"])
    labels = sorted(frame["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    frame["labels"] = frame["label"].map(label2id)
    dataset = Dataset.from_pandas(frame[["text", "labels"]]).train_test_split(test_size=0.2, seed=42)
    base = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels), label2id=label2id, id2label=id2label
    )
    tokenized = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)
    arguments = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
