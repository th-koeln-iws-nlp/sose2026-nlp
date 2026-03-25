import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import spacy

    return mo, pd


@app.cell
def _(mo):
    with open("assignments/01_assignment/01_assignment.md") as f:
        assignment_text = f.read()
    mo.md(assignment_text)
    return


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/all_songs_data_enriched.csv")
    return (lyrics_df,)


@app.cell
def _(lyrics_df, mo):
    mo.ui.table(lyrics_df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
