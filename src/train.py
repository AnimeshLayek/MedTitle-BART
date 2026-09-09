"""
MedTitle-BART training script.

This script reproduces the core training pipeline used in the project:
CORD-19 metadata -> clean data -> 80/10/10 split -> BART fine-tuning.

Example:
    python src/train.py --data-dir "D:/path/to/2020-05-01"
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BART on CORD-19.")
    parser.add_argument("--data-dir", required=True, help="Directory containing metadata.csv")
    parser.add_argument("--output-dir", default="./models/bart-cord19-final")
    parser.add_argument("--base-model", default="facebook/bart-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--input-max-length", type=int, default=384)
    parser.add_argument("--target-max-length", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    print(f"Loading dataset: {metadata_path}")
    df = pd.read_csv(metadata_path, low_memory=False)

    required = {"title", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dataset = df[["title", "abstract"]].rename(
        columns={"abstract": "input", "title": "output"}
    ).copy()

    dataset = dataset.dropna(subset=["input", "output"])
    dataset["input"] = dataset["input"].astype(str).str.strip()
    dataset["output"] = dataset["output"].astype(str).str.strip()
    dataset = dataset[(dataset["input"] != "") & (dataset["output"] != "")]
    dataset = dataset.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)

    train_data, temp_data = train_test_split(
        dataset, test_size=0.20, random_state=args.seed
    )
    validation_data, test_data = train_test_split(
        temp_data, test_size=0.50, random_state=args.seed
    )

    print(f"Training:   {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test:       {len(test_data)}")

    train_dataset = Dataset.from_pandas(train_data[["input", "output"]], preserve_index=False)
    validation_dataset = Dataset.from_pandas(
        validation_data[["input", "output"]], preserve_index=False
    )

    tokenizer = BartTokenizer.from_pretrained(args.base_model)

    def tokenize_data(batch):
        inputs = tokenizer(
            batch["input"],
            max_length=args.input_max_length,
            truncation=True,
        )
        targets = tokenizer(
            text_target=batch["output"],
            max_length=args.target_max_length,
            truncation=True,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_train = train_dataset.map(
        tokenize_data, batched=True, remove_columns=["input", "output"]
    )
    tokenized_validation = validation_dataset.map(
        tokenize_data, batched=True, remove_columns=["input", "output"]
    )

    model = BartForConditionalGeneration.from_pretrained(args.base_model)
    model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(os.path.dirname(args.output_dir), "training-results"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        optim="adamw_torch",
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to: {os.path.abspath(args.output_dir)}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    main()
