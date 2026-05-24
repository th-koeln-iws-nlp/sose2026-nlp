import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from torch.utils.data import Dataset

    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Dataset,
        Trainer,
        TrainingArguments,
        classification_report,
        f1_score,
        go,
        mo,
        np,
        pd,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 05


This assignment covers BERT pre-training and fine-tuning. You will fine-tune two encoder models for genre classification and compare them against each other and against the TF-IDF baseline from assignment 04.

---

1. **Fine-tune ModernBERT.** Fine-tune `answerdotai/ModernBERT-base` on the genre training split using the Hugging Face `Trainer`. Use the same `LyricsDataset` class and training setup as in the session notebook. Report train and test weighted F1 after training.

2. **Fine-tune a second BERT model.** Search [huggingface.co/models](https://huggingface.co/models) for an English encoder model of your choice. Fine-tune it on the same genre split with the same hyperparameters as task 2. Briefly explain why you picked this model.

3. **Comparison.** Plot a grouped bar chart showing F1 per genre for both fine-tuned models. Also show the overall weighted F1 for both models in a second chart. Compare these results to your best weighted F1 from assignment 04. Answer the following questions in a markdown cell:
   - Which genres benefit most from fine-tuned BERT models compared to TF-IDF?
   - Does ModernBERT outperform your chosen model? Where does each model struggle?
   - What could explain any differences you see between the two BERT models?

    """)
    return


@app.cell
def _(pd, train_test_split):
    df_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df = df_raw.dropna(subset=["lyrics", "genre"]).copy()

    genre_counts = df["genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= 200].index
    df = df[df["genre"].isin(valid_genres)].reset_index(drop=True)

    GENRES = sorted(df["genre"].unique())
    label2id = {g: i for i, g in enumerate(GENRES)}
    id2label = {i: g for g, i in label2id.items()}
    df["label"] = df["genre"].map(label2id)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["lyrics"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    print(
        f"{len(df)} songs | {len(GENRES)} genres | Train: {len(train_texts)} | Test: {len(test_texts)}"
    )
    print(genre_counts[valid_genres].to_string())
    return id2label, label2id, test_labels, test_texts, train_labels, train_texts


if __name__ == "__main__":
    app.run()
