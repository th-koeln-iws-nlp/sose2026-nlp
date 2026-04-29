import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    return TfidfVectorizer, cosine_similarity, mo, np, pd


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    print(f"{len(lyrics_df)} songs")
    return (lyrics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 04

    This assignment covers text vectorization (Session 4) and text classification (Session 5). You will build a genre classifier for song lyrics and evaluate it properly.

    ---

    1. **TF-IDF by artist.** Fit a `TfidfVectorizer` on all lyrics in the dataset (use `max_features=10000` and `stop_words='english'`). Pick 3 artists who each have at least 3 songs. For each artist, compute the **mean TF-IDF vector** across their songs and show the top 10 highest-scoring terms. Do the terms match your expectations? Briefly comment on what the characteristic vocabulary tells you about each artist's style.

    2. **Song similarity search.** Using your TF-IDF matrix from task 1, build a function `find_similar(song_id: int, n: int = 5) -> DataFrame` that takes the index of a song in the dataset and returns the `n` most similar songs by cosine similarity, excluding the query song itself. Show columns: title, artist, year, genre, lyrics, and similarity score. Run it for 3 songs of your choice. Do the results make sense?

    3. **Six classifiers.** Filter the dataset to genres with at least 200 songs. Split into 80% train / 20% test using `train_test_split` with `stratify=y` and `random_state=42`. Train the following 6 models and evaluate on the test set:

       | | Unigrams only | Unigrams + Bigrams + Trigrams |
       |---|---|---|
       | **Multinomial Naive Bayes** | Model 1 | Model 2 |
       | **Logistic Regression** | Model 3 | Model 4 |
       | **Linear SVC** | Model 5 | Model 6 |

       Use `TfidfVectorizer` with `max_features=10000` and `stop_words='english'`. For unigrams use `ngram_range=(1,1)`, for the trigram variant use `ngram_range=(1,3)`.

       Present all results in a single comparison table with columns: model name, ngram range, accuracy, macro F1, and weighted F1.

    4. **Error analysis.** For your best-performing model from task 3:

       a) Print the full `classification_report`. Which genre has the highest F1? Which has the lowest? Why might some genres be harder to classify than others?

       b) Plot a **confusion matrix** (use `ConfusionMatrixDisplay`). Which genre pair is most commonly confused? Pick 2 misclassified songs from this confused pair, print their lyrics (first 500 characters), and explain why the model might have struggled.

    5. **(Optional) Beat the baseline.** Try to improve on your best model from task 3. You can experiment with different hyperparameters (e.g., `C` for LogReg/SVC, `alpha` for NB), feature engineering (e.g., `min_df`, `max_df`, sublinear TF), preprocessing, or any other approach you like. Report what you tried and whether it helped.
    """)
    return


if __name__ == "__main__":
    app.run()
