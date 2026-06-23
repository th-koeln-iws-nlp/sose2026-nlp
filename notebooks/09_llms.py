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
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import time
    import pandas as pd
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from openai import OpenAI

    return OpenAI, genai, load_dotenv, os, pd, time, types


@app.cell
def _(genai, load_dotenv, os):
    load_dotenv()
    _api_key = os.getenv("GEMINI_API_KEY")
    gen_client = genai.Client(api_key=_api_key, vertexai=True)
    return (gen_client,)


@app.cell
def _(OpenAI, os):
    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
    )
    owui_models = [m.id for m in owui_client.models.list().data]
    return owui_client, owui_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 9: Working with LLMs

    We go from a raw API call to a working genre classifier, step by step.
    Every call in this notebook goes through one function -- `call_model` --
    and every token spent is logged to the usage table at the bottom.
    This is the same setup you will reuse in Assignment 06.
    """)
    return


@app.cell
def _():
    GEMINI_MODEL = "gemini-3-flash-preview"

    # Gemini 3 Flash Preview pricing, June 2026 (EUR per 1M tokens)
    # Self-hosted models cost 0 -- update here if Gemini pricing changes
    GEMINI_INPUT_PRICE_PER_1M = 0.50
    GEMINI_OUTPUT_PRICE_PER_1M = 3.00

    LYRICS_CHAR_LIMIT = 1500
    return (
        GEMINI_INPUT_PRICE_PER_1M,
        GEMINI_MODEL,
        GEMINI_OUTPUT_PRICE_PER_1M,
        LYRICS_CHAR_LIMIT,
    )


@app.cell
def _(pd):
    billboard_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    return (billboard_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allowed Genre Labels

    We keep only genres with at least 200 songs in the dataset.
    This is the same label set as Assignment 06 -- your classifier must output
    exactly one of these labels.
    """)
    return


@app.cell
def _(billboard_df, mo):
    genre_counts = billboard_df["genre"].value_counts()
    allowed_genres = sorted(genre_counts[genre_counts >= 200].index.tolist())
    filtered_df = billboard_df[billboard_df["genre"].isin(allowed_genres)].copy()
    filtered_df["label"] = filtered_df.apply(
        lambda r: f"{r['artist']} - {r['song_title']} [{r['genre']}]", axis=1
    )

    mo.callout(
        mo.md(
            f"**Label set ({len(allowed_genres)} genres):** {', '.join(allowed_genres)}\n\n"
            f"{len(filtered_df)} songs across these genres."
        ),
        kind="info",
    )
    return allowed_genres, filtered_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup

    Pick a provider and model. All sections below use this selection.
    Switch the provider and the model dropdown updates automatically.
    """)
    return


@app.cell
def _(mo):
    provider_dropdown = mo.ui.dropdown(
        options=["Gemini", "Self-hosted"],
        value="Gemini",
        label="Provider",
    )
    provider_dropdown
    return (provider_dropdown,)


@app.cell
def _(GEMINI_MODEL, mo, owui_models, provider_dropdown):
    if provider_dropdown.value == "Gemini":
        _opts = [GEMINI_MODEL]
    else:
        _opts = owui_models
    model_dropdown = mo.ui.dropdown(options=_opts, value=_opts[0], label="Model")
    model_dropdown
    return (model_dropdown,)


@app.cell
def _(GEMINI_MODEL, gen_client, time, types):
    def call_gemini(system: str, user: str, temperature: float = 0.0):
        """Returns (response_text, usage) where usage = {input, output, total, latency}."""
        t0 = time.perf_counter()
        resp = gen_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
            ),
        )
        usage = {
            "input": resp.usage_metadata.prompt_token_count or 0,
            "output": resp.usage_metadata.candidates_token_count or 0,
            "total": resp.usage_metadata.total_token_count or 0,
            "latency": time.perf_counter() - t0,
        }
        return resp.text.strip(), usage

    return (call_gemini,)


@app.cell
def _(model_dropdown, owui_client, time):
    def call_owui(system: str, user: str, temperature: float = 0.0):
        """Returns (response_text, usage) where usage = {input, output, total, latency}."""
        t0 = time.perf_counter()
        resp = owui_client.chat.completions.create(
            model=model_dropdown.value,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        usage = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
            "total": resp.usage.total_tokens,
            "latency": time.perf_counter() - t0,
        }
        return resp.choices[0].message.content.strip(), usage

    return (call_owui,)


@app.cell
def _(call_gemini, call_owui, provider_dropdown):
    def call_model(system: str, user: str, temperature: float = 0.0):
        """Dispatch to the selected backend. Returns (response_text, usage)."""
        if provider_dropdown.value == "Gemini":
            return call_gemini(system, user, temperature)
        return call_owui(system, user, temperature)

    return (call_model,)


@app.cell
def _(
    GEMINI_INPUT_PRICE_PER_1M,
    GEMINI_OUTPUT_PRICE_PER_1M,
    mo,
    pd,
    provider_dropdown,
):
    def show_usage(records: list) -> object:
        """
        Display a token-and-cost table for all recorded API calls.
        Reuse this function in Assignment 06 to report your usage.
        Each record must be a dict with keys: run, input, output, total.
        """
        if not records:
            return mo.callout(mo.md("No API calls recorded yet."), kind="warn")

        is_paid = provider_dropdown.value == "Gemini"
        agg: dict = {}
        for r in records:
            k = r["run"]
            if k not in agg:
                agg[k] = {
                    "input": 0,
                    "output": 0,
                    "total": 0,
                    "latency": 0.0,
                    "calls": 0,
                }
            agg[k]["input"] += r["input"]
            agg[k]["output"] += r["output"]
            agg[k]["total"] += r["total"]
            agg[k]["latency"] += r.get("latency", 0.0)
            agg[k]["calls"] += 1

        rows = []
        for run_label, a in agg.items():
            cost = (
                (
                    a["input"] / 1e6 * GEMINI_INPUT_PRICE_PER_1M
                    + a["output"] / 1e6 * GEMINI_OUTPUT_PRICE_PER_1M
                )
                if is_paid
                else 0.0
            )
            rows.append(
                {
                    "Run": run_label,
                    "Calls": a["calls"],
                    "Input tokens": a["input"],
                    "Output tokens": a["output"],
                    "Total tokens": a["total"],
                    "Latency (s)": f"{a['latency']:.2f}",
                    "Cost (EUR)": f"{cost:.5f}",
                }
            )

        grand_in = sum(r["input"] for r in records)
        grand_out = sum(r["output"] for r in records)
        grand_all = sum(r["total"] for r in records)
        grand_latency = sum(r.get("latency", 0.0) for r in records)
        grand_cost = (
            (
                grand_in / 1e6 * GEMINI_INPUT_PRICE_PER_1M
                + grand_out / 1e6 * GEMINI_OUTPUT_PRICE_PER_1M
            )
            if is_paid
            else 0.0
        )
        rows.append(
            {
                "Run": "TOTAL",
                "Calls": sum(a["calls"] for a in agg.values()),
                "Input tokens": grand_in,
                "Output tokens": grand_out,
                "Total tokens": grand_all,
                "Latency (s)": f"{grand_latency:.2f}",
                "Cost (EUR)": f"{grand_cost:.5f}",
            }
        )

        return mo.ui.table(pd.DataFrame(rows))

    return (show_usage,)


@app.cell
def _(filtered_df, mo):
    song_dropdown = mo.ui.dropdown(
        options=dict(zip(filtered_df["label"], filtered_df.index)),
        value=filtered_df["label"].iloc[0],
        label="Pick a song",
    )
    song_dropdown
    return (song_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Raw Call: Text In, Text Out

    Minimal prompt: `"Genre, one word:"` followed by the first 1,500 characters of lyrics.
    No role, no constraints. Let's see what comes back and how many tokens it cost.
    """)
    return


@app.cell
def _(mo):
    raw_call_button = mo.ui.run_button(label="Send")
    raw_call_button
    return (raw_call_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    call_model,
    filtered_df,
    mo,
    raw_call_button,
    song_dropdown,
):
    raw_call_output = mo.md("_Click Send to make your first API call._")
    usage_raw_call = []

    if raw_call_button.value and song_dropdown.value is not None:
        _row = filtered_df.loc[song_dropdown.value]
        _lyrics = str(_row["lyrics"])[:LYRICS_CHAR_LIMIT]
        _text, _u = call_model(system="", user=f"Genre, one word:\n\n{_lyrics}")
        usage_raw_call = [{"run": "raw call", **_u}]
        raw_call_output = mo.vstack(
            [
                mo.md(f"**Model response:** `{_text}`"),
                mo.md(
                    f"**Tokens used:** input {_u['input']}, "
                    f"output {_u['output']}, total {_u['total']}"
                ),
                mo.callout(
                    mo.md(
                        "Text in, text out. The model returned something, "
                        "but we cannot rely on its format in code."
                    ),
                    kind="info",
                ),
            ]
        )

    raw_call_output
    return (usage_raw_call,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Unstructured Prompt: The Prose Failure

    We ask a better question -- "What genre is this song?" -- but give the model
    no structure. It answers in prose. Even `.strip().lower()` leaves us with
    something that does not match any allowed label.
    """)
    return


@app.cell
def _(mo):
    unstructured_button = mo.ui.run_button(label="Send")
    unstructured_button
    return (unstructured_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    allowed_genres,
    call_model,
    filtered_df,
    mo,
    song_dropdown,
    unstructured_button,
):
    unstructured_output = mo.md("_Click Send to run._")
    usage_unstructured = []

    if unstructured_button.value and song_dropdown.value is not None:
        _row = filtered_df.loc[song_dropdown.value]
        _lyrics = str(_row["lyrics"])[:LYRICS_CHAR_LIMIT]
        _raw, _u = call_model(
            system="", user=f"What genre is this song?\n\n{_lyrics}"
        )
        usage_unstructured = [{"run": "unstructured", **_u}]
        _parsed = _raw.strip().lower()
        _valid = _parsed in [g.lower() for g in allowed_genres]
        unstructured_output = mo.vstack(
            [
                mo.md(f"**Raw response:**\n\n> {_raw}"),
                mo.md(f"**After `.strip().lower()`:** `{_parsed}`"),
                mo.callout(
                    mo.md(
                        "This response does not match any allowed label. "
                        "It is not usable by a program."
                        if not _valid
                        else "This happened to be a valid label, "
                        "but the output format is not guaranteed."
                    ),
                    kind="danger" if not _valid else "warn",
                ),
            ]
        )

    unstructured_output
    return (usage_unstructured,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## System Prompt: Constrained Classification

    We add a system prompt that gives the model a role, the exact list of
    allowed labels, and the rule: answer with exactly one label, nothing else.
    We validate the output and retry once if the model goes off-label.
    """)
    return


@app.cell
def _(allowed_genres):
    SYSTEM_PROMPT = (
        "You are a music genre classifier.\n"
        "Classify the song lyrics into exactly one of these genres: "
        + ", ".join(allowed_genres)
        + ".\nRespond with exactly one genre label from the list above and nothing else."
    )
    return (SYSTEM_PROMPT,)


@app.cell
def _(SYSTEM_PROMPT, mo):
    classify_button = mo.ui.run_button(label="Classify")
    mo.vstack(
        [
            mo.accordion(
                {"View system prompt": mo.md(f"```\n{SYSTEM_PROMPT}\n```")}
            ),
            classify_button,
        ]
    )
    return (classify_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    SYSTEM_PROMPT,
    allowed_genres,
    call_model,
    classify_button,
    filtered_df,
    mo,
    song_dropdown,
):
    classify_output = mo.md("_Click Classify to run._")
    usage_system_prompt = []

    if classify_button.value and song_dropdown.value is not None:
        _row = filtered_df.loc[song_dropdown.value]
        _lyrics = str(_row["lyrics"])[:LYRICS_CHAR_LIMIT]
        _true_genre = _row["genre"]
        _allowed_lower = [g.lower() for g in allowed_genres]

        _pred, _u = call_model(system=SYSTEM_PROMPT, user=_lyrics)
        usage_system_prompt = [{"run": "system prompt", **_u}]

        if _pred.lower() not in _allowed_lower:
            _pred2, _u2 = call_model(system=SYSTEM_PROMPT, user=_lyrics)
            usage_system_prompt.append({"run": "system prompt", **_u2})
            _pred = _pred2 if _pred2.lower() in _allowed_lower else "unknown"

        _retried = len(usage_system_prompt) > 1
        _correct = _pred.lower() == _true_genre.lower()
        classify_output = mo.vstack(
            [
                mo.md(
                    f"**Prediction:** `{_pred}` | "
                    f"**True genre:** `{_true_genre}`"
                    + (" | _retried once_" if _retried else "")
                ),
                mo.callout(
                    mo.md(
                        "Correct."
                        if _correct
                        else f"Wrong -- true genre is **{_true_genre}**."
                    ),
                    kind="success" if _correct else "danger",
                ),
            ]
        )

    classify_output
    return (usage_system_prompt,)


@app.cell
def _(show_usage, usage_raw_call, usage_system_prompt, usage_unstructured):
    show_usage(usage_raw_call + usage_unstructured + usage_system_prompt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Few-shot Classification

    We prepend two labeled examples per genre before the lyrics.
    The model now sees what the output should look like before it classifies.
    """)
    return


@app.cell
def _(filtered_df):
    few_shot_pool = (
        filtered_df.groupby("genre")
        .sample(n=2, random_state=99)
        .reset_index(drop=True)
    )
    return (few_shot_pool,)


@app.cell
def _(mo):
    few_shot_button = mo.ui.run_button(label="Classify (few-shot)")
    few_shot_button
    return (few_shot_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    SYSTEM_PROMPT,
    call_model,
    few_shot_button,
    few_shot_pool,
    filtered_df,
    mo,
    song_dropdown,
):
    few_shot_output = mo.md("_Click to run._")
    usage_few_shot = []

    if few_shot_button.value and song_dropdown.value is not None:
        _row = filtered_df.loc[song_dropdown.value]
        _lyrics = str(_row["lyrics"])[:LYRICS_CHAR_LIMIT]

        _examples = "\n\n".join(
            f"Lyrics:\n{str(ex['lyrics'])[:LYRICS_CHAR_LIMIT]}\nGenre: {ex['genre']}"
            for _, ex in few_shot_pool.iterrows()
        )
        print(_examples)
        _pred, _u = call_model(
            system=SYSTEM_PROMPT, user=f"{_examples}\n\nLyrics:\n{_lyrics}\nGenre:"
        )
        usage_few_shot = [{"run": "few-shot", **_u}]

        _correct = _pred.lower() == _row["genre"].lower()
        few_shot_output = mo.vstack(
            [
                mo.md(
                    f"**Prediction:** `{_pred}` | **True genre:** `{_row['genre']}`"
                ),
                mo.callout(
                    mo.md(
                        "Correct."
                        if _correct
                        else f"Wrong -- true genre is **{_row['genre']}**."
                    ),
                    kind="success" if _correct else "danger",
                ),
            ]
        )

    few_shot_output
    return (usage_few_shot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Temperature: How Decoding Affects Output

    We send the same song and system prompt three times at different temperatures:
    0, 0.7, and 1.2. Temperature 0 is deterministic. Higher values make the model
    hedge, add words, or drift off the allowed label set entirely.

    Takeaway: classification tasks always want temperature 0.
    """)
    return


@app.cell
def _(mo):
    temperature_button = mo.ui.run_button(label="Run at all three temperatures")
    temperature_button
    return (temperature_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    SYSTEM_PROMPT,
    call_model,
    filtered_df,
    mo,
    pd,
    song_dropdown,
    temperature_button,
):
    temperature_output = mo.md("_Click the button to run._")
    usage_temperature = []

    if temperature_button.value and song_dropdown.value is not None:
        _row = filtered_df.loc[song_dropdown.value]
        _lyrics = str(_row["lyrics"])[:LYRICS_CHAR_LIMIT]

        temperature_rows = []
        for _temp in [0.0, 0.7, 1.2]:
            _pred, _u = call_model(
                system=SYSTEM_PROMPT, user=_lyrics, temperature=_temp
            )
            usage_temperature.append({"run": f"temp={_temp}", **_u})
            temperature_rows.append({"Temperature": _temp, "Prediction": _pred})

        temperature_output = mo.vstack(
            [
                mo.ui.table(pd.DataFrame(temperature_rows)),
                mo.callout(
                    mo.md(
                        "Temperature 0 minimizes randomness and is the most reproducible setting. "
                        "In practice, GPU batch processing means results can still vary slightly "
                        "between runs even at temperature 0. "
                        "Higher temperatures add deliberate randomness on top of that -- "
                        "the model may produce invalid labels or add extra text."
                    ),
                    kind="info",
                ),
            ]
        )

    temperature_output
    return (usage_temperature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Usage Report

    Every API call above is counted here, broken out by section.
    This is the report you will reuse in Assignment 06 to document
    your token consumption and cost for zero-shot and few-shot runs.
    """)
    return


@app.cell
def _(
    show_usage,
    usage_few_shot,
    usage_raw_call,
    usage_system_prompt,
    usage_temperature,
    usage_unstructured,
):
    all_records = (
        usage_raw_call
        + usage_unstructured
        + usage_system_prompt
        + usage_few_shot
        + usage_temperature
    )
    show_usage(all_records)
    return


if __name__ == "__main__":
    app.run()
