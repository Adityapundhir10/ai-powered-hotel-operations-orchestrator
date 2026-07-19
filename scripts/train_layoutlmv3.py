"""Fine-tune LayoutLMv3 on a FUNSD-style hotel-document dataset.

Input directory should contain images and an annotations.jsonl file. Each JSON
record needs: image, words, boxes, labels. Boxes must use 0..1000 coordinates.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="models/layoutlmv3-checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    from datasets import Dataset
    from PIL import Image
    from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, Trainer, TrainingArguments

    data_dir = Path(args.data_dir)
    records = [json.loads(line) for line in (data_dir / "annotations.jsonl").read_text().splitlines() if line.strip()]
    labels = sorted({label for record in records for label in record["labels"]})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    base = "microsoft/layoutlmv3-base"
    processor = AutoProcessor.from_pretrained(base, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    def encode(record):
        image = Image.open(data_dir / record["image"]).convert("RGB")
        word_labels = [label2id[label] for label in record["labels"]]
        encoded = processor(
            image, record["words"], boxes=record["boxes"], word_labels=word_labels,
            truncation=True, padding="max_length"
        )
        return encoded

    dataset = Dataset.from_list(records).map(encode, remove_columns=list(records[0].keys()))
    split = dataset.train_test_split(test_size=0.2, seed=42)
    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        learning_rate=5e-5, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, report_to="none"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=split["train"], eval_dataset=split["test"])
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
