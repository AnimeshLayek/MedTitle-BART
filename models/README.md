# Trained Model

The fine-tuned BART checkpoint is not committed to the normal Git repository because model files can be large.

## Model

Base model:

`facebook/bart-base`

Fine-tuned task:

`Medical abstract → research title`

Dataset:

`CORD-19 2020-05-01`

## Test Results

- ROUGE-1: `0.445468`
- ROUGE-2: `0.240777`
- ROUGE-L: `0.389950`

## Local Model Path

The original experiment stored the trained model at:

```text
D:\code\Final Year Project\BART_MODEL\bart-cord19-final
```

For another machine, pass the model directory to `src/predict.py` or `src/evaluate.py`.

## Optional Git LFS

If you decide to publish the checkpoint through GitHub, use Git LFS rather than normal Git for large model objects.
