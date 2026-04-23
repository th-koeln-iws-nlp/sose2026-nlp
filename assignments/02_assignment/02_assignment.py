import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import spacy

    nlp = spacy.load("en_core_web_sm")
    return mo, nlp, pd


@app.cell
def _(nlp, text):
    def preprocess(
        texts: list[str],
        lowercase: bool = True,
        remove_stopwords: bool = True,
        remove_punct: bool = True,
        lemmatize: bool = True,
    ) -> list[str]:
        if lowercase:
            text = text.lower()

        docs = nlp.pipe(text, batch_size=256)
        all_docs = []
        for doc in all_docs:
            tokens = []
            for token in doc:
                if remove_punct and token.is_punct:
                    continue
                if remove_stopwords and token.is_stop:
                    continue

                if lemmatize:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
            all_docs.append(tokens)

        return all_docs

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assignment 02

    Use the `preprocess()` function from the [normalization notebook](../../notebooks/normalization.py) to tokenize, clean, and lemmatize lyrics.

    ---

    1. **Top 20 words.** Apply the preprocessing pipeline to all lyrics in the dataset. Create two bar charts side by side: the 20 most frequent words **with** stop words and the 20 most frequent words **without** stop words. What changes? Briefly comment on what you observe.

    2. **Who talks about love?** Calculate for each song the percentage of tokens that are the lemma "love". Which genre has the highest average love-percentage per song? Visualize the result as a bar chart across genres.

    3. **Vocabulary richness.** The [Pudding](https://pudding.cool/projects/vocabulary/index.html) visualized the vocabulary size of popular rappers. Let's do something similar: for each artist with at least 3 songs in the dataset, calculate the **number of unique word types** and the **type-token ratio** (unique types / total tokens) from their preprocessed lyrics. Show the top 10 artists with the largest vocabulary and the top 10 artists with the highest type-token ratio. Are they the same artists? Why might these two rankings differ?
    """)
    return


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    return (lyrics_df,)


@app.cell
def _(lyrics_df, mo):
    mo.ui.table(lyrics_df)
    return


@app.cell
def _(lyrics_df, nlp):
    docs = nlp.pipe(lyrics_df["lyrics"][:500].to_list(), batch_size=256)
    for doc in list(docs):
        print([token.text for token in doc])
    return


if __name__ == "__main__":
    app.run()
