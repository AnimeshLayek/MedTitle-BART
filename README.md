# COVID-19 Research Abstract Title Summarization Using BART

## 1. Project Overview

This final-year project develops an automatic title-generation system
for COVID-19 research papers. The system takes a research-paper abstract
as input and uses a fine-tuned BART sequence-to-sequence model to
generate a concise research-paper title.

The project uses data derived from the CORD-19 COVID-19 research dataset
and performs supervised text-to-text learning:

``` text
Research Abstract
       ↓
Data Preparation
       ↓
Train / Validation / Test Split
       ↓
BART Tokenization
       ↓
BART-base Fine-tuning
       ↓
Title Generation
       ↓
ROUGE Evaluation
       ↓
Saved Model + Results
```

------------------------------------------------------------------------

## 2. Project Objective

The main objective is to build and evaluate an NLP model that can
automatically generate meaningful titles from scientific abstracts.

### Specific objectives

1.  Prepare a COVID-19 research dataset for supervised title generation.
2.  Split the data into training, validation, and testing sets.
3.  Tokenize abstracts and titles using the BART tokenizer.
4.  Fine-tune `facebook/bart-base`.
5.  Generate titles for unseen test abstracts.
6.  Compare generated titles with reference titles.
7.  Evaluate the model using ROUGE-1, ROUGE-2, and ROUGE-L.
8.  Save the trained model for future use.
9.  Save predictions and evaluation results as CSV files.

------------------------------------------------------------------------

## 3. Problem Statement

Scientific research papers often have long and technically complex
abstracts. Creating concise and informative titles requires human effort
and domain knowledge.

This project investigates whether a pretrained Transformer model can
learn the relationship between scientific abstracts and their
corresponding titles and automatically generate suitable titles.

``` text
Input:
A scientific research abstract

Output:
A short research-paper title
```

------------------------------------------------------------------------

## 4. Dataset

The project uses COVID-19 research data derived from the CORD-19
dataset.

For this project, the important fields are:

-   `input` → research abstract
-   `output` → corresponding research-paper title

Example:

``` text
input:
OBJECTIVE: To compare maternal and perinatal outcomes ...

output:
Conservative management of early-onset severe preeclampsia...
```

These fields form the supervised input-output pairs used for model
training.

------------------------------------------------------------------------

## 5. Dataset Split

The dataset was divided into three parts.

### First split

80% training and 20% temporary data:

``` python
train_data, temp_data = train_test_split(
    dataset,
    test_size=0.20,
    random_state=42
)
```

### Second split

The temporary 20% was divided equally into validation and test sets:

``` python
validation_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42
)
```

### Final dataset sizes

``` text
Training   : 38,882 examples
Validation : 4,860 examples
Test       : 4,861 examples
```

Total:

``` text
48,603 examples
```

The fixed `random_state=42` makes the split reproducible.

------------------------------------------------------------------------

## 6. Technology Stack

### Programming language

-   Python

### Main libraries

-   PyTorch
-   Hugging Face Transformers
-   Hugging Face Datasets
-   scikit-learn
-   pandas
-   NumPy
-   ROUGE evaluation tools

### Model

``` text
facebook/bart-base
```

### Development environment

The project was developed locally using Jupyter notebooks in Visual
Studio Code.

------------------------------------------------------------------------

## 7. Hardware

The project was configured for:

``` text
GPU: NVIDIA GeForce RTX 2050
GPU VRAM: 4 GB
```

The NVIDIA GPU was detected using `nvidia-smi`.

PyTorch was also successfully configured to recognize CUDA. The
successful GPU test showed:

``` text
GPU available: True
GPU: NVIDIA GeForce RTX 2050
GPU memory: 4.0 GB
```

Because the GPU has only 4 GB VRAM, memory-conscious training and
inference settings are important.

------------------------------------------------------------------------

## 8. BART Model

The project uses the pretrained:

``` text
facebook/bart-base
```

BART is a Transformer encoder-decoder architecture suitable for text
generation.

For this project:

``` text
Encoder:
Research abstract

Decoder:
Research title
```

The pretrained model provides general language knowledge, while
fine-tuning adapts it to the scientific abstract-to-title task.

------------------------------------------------------------------------

## 9. Tokenization

The BART tokenizer converts text into token IDs that can be processed by
the neural network.

### Input configuration

Abstracts were tokenized with:

``` python
max_length=512
truncation=True
padding="max_length"
```

Therefore, each input is limited/padded to 512 tokens.

### Target configuration

Titles were tokenized with:

``` python
max_length=64
truncation=True
padding="max_length"
```

Therefore, titles are handled within a maximum length of 64 tokens.

### Tokenized dataset

After tokenization, each example contains:

``` text
input_ids
attention_mask
labels
```

The tokenized datasets contained:

``` text
Training   : 38,882 rows
Validation : 4,860 rows
Test       : 4,861 rows
```

------------------------------------------------------------------------

## 10. Labels

The title is the target output of the model.

The target title tokens are stored in:

``` text
labels
```

Padding positions are ignored during loss calculation where appropriate.

A typical transformation is:

``` python
token if token != tokenizer.pad_token_id else -100
```

The value `-100` is used so that those positions are ignored by the
training loss.

------------------------------------------------------------------------

## 11. Model Loading

The pretrained BART model was loaded using:

``` python
from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base"
)

model.config.use_cache = False
```

The model loaded successfully.

------------------------------------------------------------------------

## 12. Training

The BART model was fine-tuned using `Seq2SeqTrainer`.

The training process was:

``` text
Training dataset
       ↓
BART model
       ↓
Loss calculation
       ↓
Backpropagation
       ↓
Parameter updates
       ↓
Validation
```

The training completed successfully.

The completed training run shown during development reached:

``` text
Global steps: 4861
Epoch: 1
```

The displayed final values included approximately:

``` text
Training loss   : 17.2827
Validation loss : 1.8289
```

------------------------------------------------------------------------

## 13. Training Result

The training log showed approximately:

    Step   Training Loss   Validation Loss
  ------ --------------- -----------------
    1000       18.054718          2.074405
    2000       17.261466          1.982686
    3000       17.110913          1.919318
    4000       16.469017          1.851342
    4861       15.846259          1.828892

For the formal report, use the exact values from the final training log.

------------------------------------------------------------------------

## 14. Prediction / Inference

After training, the model was used to generate titles for the unseen
test set.

The prediction workflow was:

``` text
Test abstract
     ↓
BART tokenizer
     ↓
Fine-tuned BART model
     ↓
Text generation
     ↓
Generated title
```

Prediction was performed in batches to reduce memory usage on the 4 GB
GPU.

------------------------------------------------------------------------

## 15. Actual vs Generated Titles

Two lists are required for evaluation:

``` text
actual_titles
predicted_titles
```

`actual_titles` contains the reference titles from the test dataset.

`predicted_titles` contains titles generated by the trained BART model.

Example:

``` text
ACTUAL TITLE:
Reference research-paper title

GENERATED TITLE:
Title generated by the BART model
```

Ten or more examples should be included in the final report for
qualitative evaluation.

------------------------------------------------------------------------

## 16. ROUGE Evaluation

The project uses three automatic evaluation metrics.

### ROUGE-1

Measures overlap of individual words/unigrams between the reference and
generated title.

### ROUGE-2

Measures overlap of two-word sequences/bigrams.

### ROUGE-L

Measures similarity based on the Longest Common Subsequence (LCS).

Higher ROUGE scores generally indicate greater textual overlap between
generated and reference titles.

The final evaluation results were saved to:

``` text
rouge_results.csv
```

### Final ROUGE results

The final evaluation results are:

| Metric | Score | Percentage |
|---|---:|---:|
| **ROUGE-1** | **0.445468** | **44.55%** |
| **ROUGE-2** | **0.240777** | **24.08%** |
| **ROUGE-L** | **0.389950** | **38.99%** |

**Report-ready statement:**

> The fine-tuned BART model achieved a ROUGE-1 score of 0.4455 (44.55%), a ROUGE-2 score of 0.2408 (24.08%), and a ROUGE-L score of 0.3900 (38.99%) on the test dataset.

------------------------------------------------------------------------

## 17. Prediction Results

Generated predictions were saved to:

``` text
bart_predictions.csv
```

This file can be used for:

-   qualitative analysis
-   manual inspection
-   error analysis
-   final report tables
-   future experiments

A useful report table is:

  Abstract          Reference Title   Generated Title
  ----------------- ----------------- -----------------
  Test abstract 1   Reference 1       Generated 1
  Test abstract 2   Reference 2       Generated 2
  Test abstract 3   Reference 3       Generated 3

------------------------------------------------------------------------

## 18. Final Model

The trained model and tokenizer were saved successfully.

Model directory:

``` text
D:\code\Final Year Project\BART_MODEL\bart-cord19-final
```

The model was saved using:

``` python
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
```

This directory should be preserved because it contains the trained model
required for future inference.

------------------------------------------------------------------------

## 19. Project Folder Structure

The project currently contains folders/files similar to:

``` text
Final Year Project/
│
├── BART_MODEL/
│   ├── bart-cord19-final/
│   │   ├── model files
│   │   ├── tokenizer files
│   │   └── configuration files
│   │
│   ├── bart-cord19-results/
│   ├── bart-test/
│   ├── bart_predictions.csv
│   ├── rouge_results.csv
│   ├── BartCode.ipynb
│   └── BartCode2.ipynb
│
├── BIO-BART_MODEL/
│
├── cord-19_2020-05-01/
│
├── Research Paper/
│
├── cord-19_2020-05-01.tar...
│
└── setup.ipynb
```

Exact checkpoint contents may vary according to the training
configuration.

------------------------------------------------------------------------

## 20. Complete Project Workflow

### Stage 1 --- Import libraries

Load Python, PyTorch, Transformers, Datasets, pandas, and evaluation
libraries.

### Stage 2 --- Load dataset

Load the prepared COVID-19 dataset.

### Stage 3 --- Verify dataset

Check:

``` text
Number of rows
Column names
Missing values
Sample abstract
Sample title
```

### Stage 4 --- Split dataset

Create:

``` text
Training
Validation
Testing
```

### Stage 5 --- Load tokenizer

Load the BART tokenizer associated with `facebook/bart-base`.

### Stage 6 --- Tokenize data

Convert abstracts and titles into:

``` text
input_ids
attention_mask
labels
```

### Stage 7 --- Check GPU

Verify:

``` python
torch.cuda.is_available()
```

Expected project hardware:

``` text
NVIDIA GeForce RTX 2050
4 GB VRAM
```

### Stage 8 --- Load BART

Load:

``` text
facebook/bart-base
```

### Stage 9 --- Configure trainer

Create the training configuration and `Seq2SeqTrainer`.

### Stage 10 --- Fine-tune

Run:

``` python
trainer.train()
```

### Stage 11 --- Save model

Save:

``` text
bart-cord19-final
```

### Stage 12 --- Generate predictions

Run the trained model on the test dataset.

### Stage 13 --- Decode predictions

Convert generated token IDs back into text.

### Stage 14 --- Compare results

Compare:

``` text
Actual title
vs
Generated title
```

### Stage 15 --- Calculate ROUGE

Calculate:

``` text
ROUGE-1
ROUGE-2
ROUGE-L
```

### Stage 16 --- Save results

Save:

``` text
bart_predictions.csv
rouge_results.csv
```

------------------------------------------------------------------------

## 21. Common Errors Encountered

### `ModuleNotFoundError: No module named 'sklearn'`

Cause: scikit-learn was missing from the active environment.

Solution:

``` bash
pip install scikit-learn
```

Then restart the notebook kernel.

------------------------------------------------------------------------

### `NameError: name 'small_train' is not defined`

Cause: code referenced variables that had not been created.

Solution: use the actual tokenized datasets, for example:

``` python
train_dataset=tokenized_train
eval_dataset=tokenized_validation
```

------------------------------------------------------------------------

### `Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

Cause: Transformers API differences between versions.

Solution: use the compatible API for the installed Transformers version,
such as `processing_class=tokenizer` when supported.

------------------------------------------------------------------------

### `TrainingArguments object has no attribute 'generation_config'`

Cause: Transformers version/API compatibility issue.

Solution: use a compatible `Seq2SeqTrainingArguments` configuration and
avoid unsupported attributes.

------------------------------------------------------------------------

### `OverflowError: can't convert negative int to unsigned`

Cause: labels contained `-100`, which is correct for loss masking but
cannot be passed directly to `batch_decode`.

Solution:

``` python
clean_labels = [
    [
        token if token != -100 else tokenizer.pad_token_id
        for token in label
    ]
    for label in labels
]

actual_titles = tokenizer.batch_decode(
    clean_labels,
    skip_special_tokens=True
)
```

------------------------------------------------------------------------

### `ValueError: expected sequence of length ...`

Cause: tokenized examples had inconsistent lengths when directly
converted into a tensor.

Solution: use tokenizer padding/data collator or pad the batch before
creating tensors.

------------------------------------------------------------------------

### `NameError: name 'actual_titles' is not defined`

Cause: the code attempted to display `actual_titles` before creating it.

Solution:

``` python
actual_titles = tokenizer.batch_decode(
    clean_labels,
    skip_special_tokens=True
)
```

Then:

``` python
print(actual_titles[i])
```

------------------------------------------------------------------------

## 22. Reproducibility

The dataset split uses:

``` python
random_state=42
```

For complete reproducibility, record:

-   Python version
-   PyTorch version
-   Transformers version
-   Datasets version
-   scikit-learn version
-   CUDA version
-   NVIDIA driver version
-   model name
-   tokenizer name
-   training arguments
-   random seeds

------------------------------------------------------------------------

## 23. Loading the Saved Model

The saved model can be loaded without retraining:

``` python
from transformers import BartForConditionalGeneration, BartTokenizer

model_path = r"D:\code\Final Year Project\BART_MODEL\bart-cord19-final"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
```

Example inference:

``` python
abstract = """
Paste a COVID-19 research abstract here.
"""

inputs = tokenizer(
    abstract,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=64,
    num_beams=4,
    early_stopping=True
)

title = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("Generated Title:")
print(title)
```

For GPU inference, move the model and input tensors to CUDA after
confirming CUDA is available.

------------------------------------------------------------------------

## 24. Qualitative Evaluation

ROUGE scores should be supported by manual inspection.

Check whether each generated title:

1.  Represents the main topic of the abstract.
2.  Preserves important medical/scientific terminology.
3.  Is grammatically correct.
4.  Is concise.
5.  Avoids irrelevant information.
6.  Does not invent unsupported claims.
7.  Preserves the main meaning of the reference title.

------------------------------------------------------------------------

## 25. Error Analysis

Possible model errors include:

### Missing important terms

The model may omit an important disease, treatment, population, or
methodology term.

### Over-generalization

The generated title may be too broad.

### Repetition

The model may repeat words or phrases.

### Hallucination

The model may generate information that is not supported by the
abstract.

### Incomplete title

The generated title may be grammatically incomplete.

### Reference mismatch

A generated title may be semantically reasonable but use different
wording from the human-written title, producing a lower ROUGE score.

------------------------------------------------------------------------

## 26. Limitations

### Hardware limitation

The model was trained on a laptop with a 4 GB RTX 2050 GPU. This limits
batch size and training configuration options.

### Input length

Abstracts longer than 512 BART tokens are truncated.

### Target length

Titles are limited to 64 tokens during tokenization/generation.

### Dataset limitation

The model is specialized for the data distribution used during training
and may not perform equally well on unrelated scientific domains.

### ROUGE limitation

ROUGE measures textual overlap. A generated title can be semantically
good while receiving a lower ROUGE score if it uses different wording.

------------------------------------------------------------------------

## 27. Future Improvements

Possible future work:

1.  Train for additional epochs if resources permit.
2.  Tune the learning rate.
3.  Experiment with batch size and gradient accumulation.
4.  Compare BART with T5, PEGASUS, or other summarization models.
5.  Use larger pretrained models if sufficient GPU memory is available.
6.  Improve preprocessing and data cleaning.
7.  Remove duplicate or low-quality records.
8.  Perform domain-specific evaluation by medical experts.
9.  Add semantic metrics such as BERTScore.
10. Build a web interface for title generation.
11. Deploy the trained model as an API.
12. Compare generated titles with multiple reference titles where
    available.

------------------------------------------------------------------------

## 28. Final Deliverables

The major project deliverables are:

### Trained model

``` text
bart-cord19-final/
```

### Prediction file

``` text
bart_predictions.csv
```

### Evaluation file

``` text
rouge_results.csv
```

### Source notebooks

``` text
BartCode.ipynb
BartCode2.ipynb
```

### Documentation

``` text
README.md
```

### Research/report material

``` text
Research Paper/
```

------------------------------------------------------------------------

## 29. Final Submission Checklist

-   [x] Dataset loaded successfully
-   [x] Dataset split completed
-   [x] Tokenizer loaded
-   [x] Tokenization completed
-   [x] CUDA/GPU tested
-   [x] BART-base loaded
-   [x] Training completed
-   [x] Model saved
-   [x] Test predictions generated
-   [x] Predictions saved to CSV
-   [x] ROUGE calculated
-   [x] ROUGE results saved
-   [ ] Final ROUGE scores inserted into report
-   [ ] Training-loss graph prepared
-   [ ] Validation-loss graph prepared
-   [ ] Actual vs generated examples selected
-   [ ] Error analysis completed
-   [ ] Final report written
-   [ ] Saved model backed up

------------------------------------------------------------------------

## 30. Suggested Final Report Structure

``` text
Chapter 1 — Introduction
    1.1 Background
    1.2 Problem Statement
    1.3 Motivation
    1.4 Objectives
    1.5 Scope

Chapter 2 — Literature Review
    2.1 Natural Language Processing
    2.2 Text Summarization
    2.3 Transformer Architecture
    2.4 BART
    2.5 Scientific Text Summarization
    2.6 COVID-19 / CORD-19 Dataset
    2.7 ROUGE Evaluation

Chapter 3 — Methodology
    3.1 Dataset
    3.2 Data Preparation
    3.3 Dataset Split
    3.4 Tokenization
    3.5 BART Model
    3.6 Training
    3.7 Prediction
    3.8 Evaluation

Chapter 4 — Implementation
    4.1 Software Environment
    4.2 Hardware
    4.3 Dataset Loading
    4.4 Tokenization
    4.5 Model Training
    4.6 Model Saving
    4.7 Prediction
    4.8 ROUGE Calculation

Chapter 5 — Results and Discussion
    5.1 Training Results
    5.2 Validation Results
    5.3 ROUGE Results
    5.4 Actual vs Generated Titles
    5.5 Error Analysis
    5.6 Discussion

Chapter 6 — Conclusion and Future Work
    6.1 Conclusion
    6.2 Limitations
    6.3 Future Work

References
Appendix
```

------------------------------------------------------------------------

## 31. Conclusion

This project implements an end-to-end scientific title-generation system
using a fine-tuned BART Transformer model.

The workflow begins with COVID-19 research abstracts and corresponding
titles, prepares and splits the dataset, tokenizes the text, fine-tunes
`facebook/bart-base`, generates titles for unseen test abstracts, and
evaluates the generated titles using ROUGE-1, ROUGE-2, and ROUGE-L.

The trained model has been successfully saved at:

``` text
D:\code\Final Year Project\BART_MODEL\bart-cord19-final
```

Prediction and evaluation outputs have also been saved at:

``` text
D:\code\Final Year Project\BART_MODEL\bart_predictions.csv

D:\code\Final Year Project\BART_MODEL\rouge_results.csv
```

The project is therefore a complete prototype of an automatic COVID-19
scientific abstract-to-title generation system. The remaining work is
mainly results analysis, visualization, documentation, and final report
preparation.

------------------------------------------------------------------------

## 32. Important Note About Reported Results

Do not place invented performance numbers in the final report.

The exact ROUGE-1, ROUGE-2, and ROUGE-L values must be copied from:

``` text
rouge_results.csv
```

Training and validation loss values should be taken from the final
training log.

This README documents the project architecture, workflow, implementation
decisions, hardware, files, errors encountered, evaluation process, and
future work.
