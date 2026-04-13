import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assignment 01

    To count words in this assignment, use the string `split()` method in Python. Next week we gonna look at a more precise way using spaCy.

    ---

    1. What is the average length (number of words) of song lyrics per genre? Visualize this as a bar chart.

    2. Plot a timeline for the average lyrics length (number of words) for each year.

    3. Which song has the least amount of unique words? (*Tip: For scenarios like this, there is the very handy `set` construction in Python.*)

    4. Who are the fastest rappers? List the top 10 songs with the highest number of words to song duration ratio. Is there a song in the top 10 which does not have the genre label "Hip-Hop"?
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


if __name__ == "__main__":
    app.run()
