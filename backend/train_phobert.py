import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = os.getenv("MODEL_NAME", "vinai/phobert-base")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data/processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./checkpoints"))
BEST_MODEL_DIR = OUTPUT_DIR / "best_phobert_fake_news"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    df = pd.read_csv(csv_path)

    text_col = "text" if "text" in df.columns else "content" if "content" in df.columns else None
    label_col = "label" if "label" in df.columns else "target" if "target" in df.columns else None

    if text_col is None or label_col is None:
        raise ValueError(
            f"File {csv_path} phải chứa ít nhất 2 cột: text/content và label/target. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col]).copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    normalized_mapping = {
        "fake": 1,
        "tin_gia": 1,
        "tin giả": 1,
        "gia": 1,
        "real": 0,
        "tin_that": 0,
        "tin thật": 0,
        "that": 0,
    }

    if df["label"].dtype == "object":
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(normalized_mapping)

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if set(df["label"].unique()) - {0, 1}:
        raise ValueError(f"Nhãn trong {csv_path} phải là 0/1. Các giá trị hiện có: {sorted(df['label'].unique().tolist())}")

    return df


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(DATA_DIR / "train.csv")
    val_df = load_dataset(DATA_DIR / "val.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        save_total_limit=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    print(f"Huấn luyện hoàn tất. Mô hình tốt nhất được lưu tại: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
