"""
MedTitle-BART evaluation script.

Generates titles for the test split and calculates ROUGE-1, ROUGE-2,
and ROUGE-L.

Example:
    python src/evaluate.py --model-dir "./models/bart-cord19-final" \
        --data-dir "D:/path/to/2020-05-01" --output-dir "./results"
"""

from __future__ import annotations

import argparse
import os
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_data(data_dir: str, seed: int):
    path = os.path.join(data_dir, "metadata.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metadata.csv not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    dataset = df[["title", "abstract"]].rename(
        columns={"abstract": "input", "title": "output"}
    ).dropna().copy()

    dataset["input"] = dataset["input"].astype(str).str.strip()
    dataset["output"] = dataset["output"].astype(str).str.strip()
    dataset = dataset[(dataset["input"] != "") & (dataset["output"] != "")]
    dataset = dataset.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)

    _, temp = train_test_split(dataset, test_size=0.20, random_state=seed)
    _, test_data = train_test_split(temp, test_size=0.50, random_state=seed)
    return test_data.reset_index(drop=True)


def generate_titles(test_data, tokenizer, model, device, batch_size=2):
    predictions = []

    for start in range(0, len(test_data), batch_size):
        texts = test_data["input"].iloc[start:start + batch_size].tolist()

        inputs = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        predictions.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )

        if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(test_data):
            print(f"Processed {min(start + batch_size, len(test_data))}/{len(test_data)}")

    return [p.strip() for p in predictions]


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedTitle-BART.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BartTokenizer.from_pretrained(args.model_dir)
    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_data = load_test_data(args.data_dir, args.seed)
    predicted_titles = generate_titles(
        test_data, tokenizer, model, device, args.batch_size
    )
    actual_titles = test_data["output"].tolist()

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predicted_titles,
        references=actual_titles,
        use_stemmer=True,
    )

    rouge_table = pd.DataFrame(
        {
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Score": [scores["rouge1"], scores["rouge2"], scores["rougeL"]],
        }
    )
    rouge_table["Percentage"] = (rouge_table["Score"] * 100).round(2)

    predictions_df = pd.DataFrame(
        {
            "input": test_data["input"].values,
            "actual_title": actual_titles,
            "generated_title": predicted_titles,
        }
    )

    rouge_path = os.path.join(args.output_dir, "rouge_results.csv")
    predictions_path = os.path.join(args.output_dir, "bart_predictions.csv")
    rouge_table.to_csv(rouge_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nROUGE RESULTS")
    print(rouge_table.to_string(index=False))
    print(f"\nSaved: {rouge_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()
