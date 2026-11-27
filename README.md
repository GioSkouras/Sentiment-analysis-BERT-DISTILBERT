# Sentiment Analysis with BERT and DistilBERT

This project compares two pretrained Transformer encoders for binary tweet
sentiment classification: **BERT** and **DistilBERT**. Each experiment has its
own notebook, and the written analysis is available in
[`BERT_DISTILBERT_analysis.pdf`](BERT_DISTILBERT_analysis.pdf).

## Notebooks

| Model | Notebook | Hugging Face checkpoint |
| --- | --- | --- |
| BERT | [`sdi2100172-hw3-bert.ipynb`](sdi2100172-hw3-bert.ipynb) | `bert-base-uncased` |
| DistilBERT | [`sdi2100172-hw3-distilbert.ipynb`](sdi2100172-hw3-distilbert.ipynb) | `distilbert-base-uncased` |

Each notebook loads raw tweet text, tokenizes it to a maximum length of 128,
fine-tunes a pretrained encoder with a classification head, evaluates on the
validation set, and predicts labels for the test set. The best checkpoint is
selected by validation accuracy.

## Model configuration

| Component | BERT | DistilBERT |
| --- | --- | --- |
| Maximum sequence length | 128 | 128 |
| Batch size | 64 | 32 |
| Training epochs | 3 | 3 |
| Optimizer | AdamW | AdamW |
| Learning rate | `3e-5` | `2e-5` |
| Weight decay | 0.01 | 0.01 |
| Random seed | 42 | 42 |

Both notebooks report validation accuracy, precision, recall, and F1 score,
and include ROC and confusion-matrix visualizations.

## Dataset format

The BERT notebook expects the Kaggle dataset at:

```text
/kaggle/input/ai-2-deep-learning-for-nlp-2025-homework-3/
```

The DistilBERT notebook expects it at:

```text
/kaggle/input/ai-2-deep-learning-for-nlp-2025-homework-3-distil-bert/
```

Each dataset directory must contain `train_dataset.csv`, `val_dataset.csv`, and
`test_dataset.csv`. Training and validation files need `Text` and `Label`
columns; the test file needs `ID` and `Text`. Labels are expected to be binary
values (`0` and `1`).

## Running the notebooks

### Kaggle

1. Open the notebook for the model you want to run.
2. Attach the corresponding dataset at the path shown above.
3. Run all cells from top to bottom. A GPU is recommended for fine-tuning.
4. Find the generated predictions in `submission.csv`.

### Local Jupyter installation

Install the main Python dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers \
    tqdm wordcloud jupyter
```

The notebooks use Kaggle-specific absolute dataset paths. Update the three
`pd.read_csv(...)` paths in the selected notebook to use your local dataset
directory. The pretrained checkpoints are downloaded from Hugging Face when
the notebook runs.

## Output

Each notebook writes `submission.csv` with the following columns:

```text
ID, Label
```

The best fine-tuned weights are also saved in the notebook's working directory
as `best_bert_model.pt` or `best_distilbert.pt`.

## Repository structure

```text
.
├── BERT_DISTILBERT_analysis.pdf       # Project analysis and discussion
├── sdi2100172-hw3-bert.ipynb          # BERT fine-tuning and inference
├── sdi2100172-hw3-distilbert.ipynb    # DistilBERT fine-tuning and inference
└── README.md                          # Project documentation
```
