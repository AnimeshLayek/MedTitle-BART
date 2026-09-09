"""
MedTitle-BART inference script.

Examples:
    python src/predict.py --model-dir "./models/bart-cord19-final"
    python src/predict.py --model-dir "./models/bart-cord19-final" --text "Medical abstract..."
"""

from __future__ import annotations

import argparse

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


def load_model(model_dir: str):
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_title(
    text: str,
    tokenizer,
    model,
    device,
    input_max_length: int = 512,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> str:
    if not text or not text.strip():
        raise ValueError("Input abstract cannot be empty.")

    inputs = tokenizer(
        text,
        max_length=input_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a medical research title.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_dir)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.text:
        print("\nGENERATED TITLE:")
        print(generate_title(args.text, tokenizer, model, device))
        return

    print("\nMedTitle-BART interactive mode")
    print("Type 'exit' to stop.")

    while True:
        text = input("\nEnter medical abstract: ").strip()
        if text.lower() == "exit":
            break
        try:
            title = generate_title(text, tokenizer, model, device)
            print("\nGENERATED TITLE:")
            print(title)
        except ValueError as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
