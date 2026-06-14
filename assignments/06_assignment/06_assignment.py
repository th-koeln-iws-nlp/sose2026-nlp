# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "google-genai",
#     "openai",
#     "python-dotenv",
#     "scikit-learn",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import plotly.express as px
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from openai import OpenAI
    from sklearn.metrics import f1_score, classification_report

    return OpenAI, classification_report, f1_score, genai, load_dotenv, mo, os, pd, px, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 06

    Keep only **genres with at least 200 songs**. From those, draw a **stratified random test sample of about 500 songs** (set a random seed so it is reproducible), and keep the remaining labeled songs as a pool to draw few-shot examples from. Those genres are your **label set**. Classify **one song per request** (we cover batching in the next session). Pick **one** model and use it throughout:

    - **Gemini 3 Flash Preview** (API), or
    - a self-hosted model from Open WebUI: **Mistral 3.2**, **Qwen 3.5**, or **Qwen 3.6**

    Set **temperature 0** everywhere (reproducible and comparable), and truncate each song's lyrics to the **first 1,500 characters** before sending. Reuse the model client setup from the session tutorial notebook.

    ---

    1. **Zero-shot genre classification.** Write a **system prompt** that turns the model into a genre classifier: give it the role, the **exact allowed genre labels** (your label set), and the instruction to answer with **one label only**. Send the truncated lyrics as the user prompt. Parse the response to a single label and **validate that it is in the allowed set** (map anything else to `unknown`, or retry the request once). Run on the test sample and report **weighted F1**.

    2. **Few-shot genre classification.** Add a small number of labeled examples (for example 2 per genre) drawn from the **example pool** to your prompt, keeping everything else identical. Re-run on the same test sample and report **weighted F1**. In a markdown cell, state whether few-shot helped and by how much.

    3. **Token and credit usage.** Report the total tokens and the cost in credits / euros for both the zero-shot and few-shot runs, the same way as in the session tutorial.

    4. **Comparison.** Plot a grouped bar chart of **F1 per genre** for your zero-shot run next to your few-shot run, and report the **overall weighted F1** for both. Also compare against your **best classifiers from the previous assignments** (TF-IDF from Assignment 04 and your best fine-tuned BERT from Assignment 05): add their overall weighted F1 to the comparison. In a markdown cell, briefly answer: which genres are hardest for the model, did few-shot improve the same genres or different ones, and how does the LLM stack up against your trained classifiers?
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df = df_raw.dropna(subset=["lyrics", "genre"]).copy()

    genre_counts = df["genre"].value_counts()
    allowed_genres = sorted(genre_counts[genre_counts >= 200].index.tolist())
    df = df[df["genre"].isin(allowed_genres)].reset_index(drop=True)

    print(f"{len(df)} songs | genres: {', '.join(allowed_genres)}")
    return allowed_genres, df


@app.cell
def _(df):
    test_df = df.groupby("genre").sample(frac=500 / len(df), random_state=42)
    pool_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Test: {len(test_df)} songs | Pool: {len(pool_df)} songs")
    return pool_df, test_df


@app.cell
def _(load_dotenv, os, genai):
    load_dotenv()
    gen_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return (gen_client,)


@app.cell
def _(OpenAI, os):
    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
    )
    owui_models = [m.id for m in owui_client.models.list().data]
    return owui_client, owui_models


if __name__ == "__main__":
    app.run()
