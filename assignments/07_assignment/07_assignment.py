# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "google-genai",
#     "openai",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import marimo as mo
    import os
    import pandas as pd
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from openai import OpenAI

    return OpenAI, ast, genai, load_dotenv, mo, os, pd, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 07

    Keep only songs with a non-empty `niche_genres` list. Draw a **random test sample of about 500 songs** (set a seed so it is reproducible), and keep the remaining labeled songs as a pool for examples and tag statistics. Pick **one** model and use it throughout:

    - **Gemini 3 Flash Preview** (API), or
    - a self-hosted model from Open WebUI: **Mistral 3.2**, **Qwen 3.5**, or **Qwen 3.6**

    Set **temperature 0**, truncate each song's lyrics to the **first 1,500 characters**, and have the model return its answer as a **JSON array of tags**, validated as in the session tutorial. One request per song.

    ---

    1. **Explore the label space.** Using the example pool, report the **number of distinct niche tags** and list the **top 50 most frequent** ones. This top 50 set is your **allowed tag set** for scoring. Report **coverage**: the share of the test sample's true tags that fall inside it.

    2. **Zero-shot prediction.** Write a **system prompt** that asks for a song's niche subgenres as a **JSON array** of up to 3 tags, from the lyrics, with no examples. Score against the true `niche_genres` with **mean Jaccard** (set overlap per song, averaged). Normalize tags to lowercase before matching.

    3. **Few-shot prediction.** Add a handful of worked examples (lyrics with their correct tags) drawn from the **example pool**, keeping everything else identical. Re-run on the same test sample and report the same metric. In a markdown cell, state whether few-shot helped and by how much.

    4. **Token and credit usage.** Report the total tokens and the cost in credits / euros for both runs, the same way as in the session tutorial.

    *Remark: Spotify's niche tags are noisy and assigned at the artist level, not the song level, so a "wrong" prediction is sometimes the model being more right than the label. Treat these scores as a rough signal, not ground truth.*

    **Optional.** Send the songs in **batches** instead of one per request, returning a JSON array keyed by song id. Re-request any ids the model drops.
    """)
    return


@app.cell
def _(ast, pd):
    df_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df_raw["niche_genres_list"] = df_raw["niche_genres"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df = (
        df_raw[df_raw["niche_genres_list"].map(len) > 0]
        .dropna(subset=["lyrics"])
        .copy()
    )
    df = df.reset_index(drop=True)
    print(f"{len(df)} songs with niche genres and lyrics")
    return (df,)


@app.cell
def _(df):
    test_df = df.sample(n=500, random_state=42)
    pool_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Test: {len(test_df)} songs | Pool: {len(pool_df)} songs")
    return pool_df, test_df


@app.cell
def _(genai, load_dotenv, os):
    load_dotenv()
    gen_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=True)
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
