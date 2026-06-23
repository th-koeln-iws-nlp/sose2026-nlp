# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "google-genai",
#     "openai",
#     "python-dotenv",
#     "pydantic",
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
    import json
    import os
    import time
    import pandas as pd
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from openai import OpenAI
    from pydantic import BaseModel

    return BaseModel, OpenAI, genai, json, load_dotenv, os, pd, time, types


@app.cell
def _(BaseModel, json):
    class Song(BaseModel):
        title: str
        lyrics: str


    def parse_song(text: str) -> Song:
        return Song.model_validate(json.loads(text))

    return Song, parse_song


@app.cell
def _(BaseModel):
    class Judgment(BaseModel):
        score: int
        reasoning: str

    return (Judgment,)


@app.cell
def _(mo):
    def render_song(song, label: str):
        return mo.md(f"**{label}**\n\n## {song.title}\n\n{song.lyrics}")

    return (render_song,)


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


@app.cell
def _():
    GEMINI_MODEL = "gemini-3-flash-preview"
    LYRICS_CHAR_LIMIT = 1500
    SONG_SYSTEM_PROMPT = "You are a creative songwriter."
    JUDGE_SYSTEM_PROMPT = "You are a music critic."
    return (
        GEMINI_MODEL,
        JUDGE_SYSTEM_PROMPT,
        LYRICS_CHAR_LIMIT,
        SONG_SYSTEM_PROMPT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 10: LLMs Continued: Structured Output and Evaluation

    Text in, text out works for reading. It breaks the moment your program needs to
    parse the result. This session covers three fixes and one bridge:

    1. **Unstructured text**: plain prose, nice to read, useless to a program
    2. **Structured output**: schema enforced by the API, validated with Pydantic
    3. **Few-shot style**: real examples teach voice and style, not just format
    4. **LLM-as-judge**: when you cannot auto-score, ask a model to score for you
    """)
    return


@app.cell
def _(pd):
    billboard_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    return (billboard_df,)


@app.cell
def _(billboard_df):
    artist_counts = billboard_df.dropna(subset=["lyrics"])["artist"].value_counts()
    songwriting_artists = sorted(artist_counts[artist_counts >= 5].index.tolist())
    return (songwriting_artists,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Pick a model, an artist to imitate, and a topic. Every step below uses these.
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
def _(mo, songwriting_artists):
    artist_dropdown = mo.ui.dropdown(
        options=songwriting_artists,
        value=songwriting_artists[0],
        label="Artist to imitate",
    )
    topic_input = mo.ui.text(
        placeholder="e.g. heartbreak, summer, roads", label="Topic"
    )
    mo.hstack([artist_dropdown, topic_input], gap=2)
    return artist_dropdown, topic_input


@app.cell
def _(GEMINI_MODEL, gen_client, time, types):
    def call_gemini(system: str, user: str, temperature: float = 0.0, schema=None):
        t0 = time.perf_counter()
        config_kwargs = dict(
            system_instruction=system or None, temperature=temperature
        )
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        resp = gen_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
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
def _(json, model_dropdown, owui_client, time):
    def call_owui(system: str, user: str, temperature: float = 0.0, schema=None):
        t0 = time.perf_counter()
        if schema is not None:
            system = (
                system
                + f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
            )
        extra = {"response_format": {"type": "json_object"}} if schema is not None else {}
        resp = owui_client.chat.completions.create(
            model=model_dropdown.value,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **extra,
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
    def call_model(system: str, user: str, temperature: float = 0.0, schema=None):
        if provider_dropdown.value == "Gemini":
            return call_gemini(system, user, temperature, schema=schema)
        return call_owui(system, user, temperature, schema=schema)

    return (call_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 1: Unstructured Text

    Prompt plainly: "Write a song in the style of {artist} about {topic}."
    The model returns readable prose or markdown. Then try `json.loads` on it.
    """)
    return


@app.cell
def _(mo):
    unstructured_button = mo.ui.run_button(label="Generate")
    unstructured_button
    return (unstructured_button,)


@app.cell
def _(artist_dropdown, call_model, json, mo, topic_input, unstructured_button):
    unstructured_output = mo.md("_Enter a topic and click Generate._")

    if unstructured_button.value and topic_input.value:
        _prompt = f"Write a song in the style of {artist_dropdown.value} about {topic_input.value}."
        _text, _u = call_model(system="", user=_prompt, temperature=0.9)

        try:
            json.loads(_text)
            _parse_result = mo.callout(
                mo.md(
                    "The model happened to return valid JSON. Try a different artist or topic to see the usual case."
                ),
                kind="warn",
            )
        except json.JSONDecodeError as e:
            _parse_result = mo.callout(
                mo.md(
                    f"`json.loads` raised:\n\n```\n{e}\n```\n\nProse is readable, but a program cannot use it."
                ),
                kind="danger",
            )

        unstructured_output = mo.vstack(
            [
                mo.md(f"**Raw response:**\n\n{_text}"),
                _parse_result,
            ]
        )

    unstructured_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 2: Demand JSON

    We pass the Pydantic schema directly to the API: Gemini via `response_schema`,
    Open WebUI via `response_format: json_object`. The API enforces the structure,
    so we get back clean JSON and validate with Pydantic.
    """)
    return


@app.cell
def _(mo):
    json_button = mo.ui.run_button(label="Generate (structured)")
    json_button
    return (json_button,)


@app.cell
def _(
    SONG_SYSTEM_PROMPT,
    Song,
    artist_dropdown,
    call_model,
    json_button,
    mo,
    parse_song,
    render_song,
    topic_input,
):
    zero_shot_song = None
    json_output = mo.md("_Enter a topic and click Generate._")

    if json_button.value and topic_input.value:
        _user = f"Write a song in the style of {artist_dropdown.value} about {topic_input.value}."
        _text, _u = call_model(
            system=SONG_SYSTEM_PROMPT, user=_user, temperature=0.9, schema=Song
        )
        try:
            zero_shot_song = parse_song(_text)
        except Exception as e:
            json_output = mo.callout(mo.md(f"Parse failed: `{e}`"), kind="danger")

        if zero_shot_song:
            json_output = mo.vstack(
                [
                    render_song(zero_shot_song, "Zero-shot"),
                    mo.callout(
                        mo.md(
                            "The API enforced the schema. Every field is typed and accessible in code."
                        ),
                        kind="success",
                    ),
                ]
            )

    json_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 3: Few-shot Style

    The zero-shot song has the right structure, but does it sound like the artist?
    We pull real songs from the dataset and inject them as `<example>` blocks.
    The model sees actual vocabulary, rhythm, and phrasing before it writes.
    """)
    return


@app.cell
def _(artist_dropdown, billboard_df):
    artist_songs = (
        billboard_df[billboard_df["artist"] == artist_dropdown.value]
        .dropna(subset=["lyrics"])
        .reset_index(drop=True)
    )
    artist_examples = artist_songs.sample(
        n=min(3, len(artist_songs)), random_state=42
    )
    return (artist_examples,)


@app.cell
def _(mo):
    few_shot_button = mo.ui.run_button(label="Generate (few-shot)")
    few_shot_button
    return (few_shot_button,)


@app.cell
def _(
    LYRICS_CHAR_LIMIT,
    SONG_SYSTEM_PROMPT,
    Song,
    artist_dropdown,
    artist_examples,
    call_model,
    few_shot_button,
    mo,
    parse_song,
    render_song,
    topic_input,
):
    few_shot_song = None
    few_shot_output = mo.md("_Enter a topic and click Generate._")

    if few_shot_button.value and topic_input.value:
        _examples_text = "\n\n".join(
            f"<example>\n{str(row['lyrics'])[:LYRICS_CHAR_LIMIT]}\n</example>"
            for _, row in artist_examples.iterrows()
        )
        _user = (
            f"Here are songs by {artist_dropdown.value}:\n\n"
            f"{_examples_text}\n\n"
            f"Write a new song in the same style about {topic_input.value}."
        )
        _text, _u = call_model(
            system=SONG_SYSTEM_PROMPT, user=_user, temperature=0.9, schema=Song
        )
        try:
            few_shot_song = parse_song(_text)
        except Exception as e:
            few_shot_output = mo.callout(
                mo.md(f"Parse failed: `{e}`"), kind="danger"
            )

        if few_shot_song:
            few_shot_output = mo.vstack(
                [
                    render_song(few_shot_song, "Few-shot"),
                    mo.callout(
                        mo.md(
                            "Few-shot examples teach voice and style, not just output format."
                        ),
                        kind="info",
                    ),
                ]
            )

    few_shot_output
    return (few_shot_song,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 4: How Do You Grade This?

    You cannot compute F1 on a generated song: there is no ground truth label.
    One proxy: ask a fresh model call to rate the output.
    We request a score from 1 to 5 and one sentence of reasoning.

    The judge has biases: it may favor longer output, more formal language, or its own
    generation style. LLM-as-judge is a useful first signal, not a substitute for human
    evaluation. This pattern comes up again in the Evaluation session.
    """)
    return


@app.cell
def _(mo):
    judge_button = mo.ui.run_button(label="Judge the few-shot song")
    judge_button
    return (judge_button,)


@app.cell
def _(
    JUDGE_SYSTEM_PROMPT,
    Judgment,
    artist_dropdown,
    call_model,
    few_shot_song,
    json,
    judge_button,
    mo,
):
    judge_output = mo.md("_Click to judge._")

    if judge_button.value:
        if few_shot_song is None:
            judge_output = mo.callout(
                mo.md("Run Step 3 first to generate a few-shot song."), kind="warn"
            )
        else:
            _song_text = f"Title: {few_shot_song.title}\n\n{few_shot_song.lyrics}"
            _judge_user = (
                f"Rate 1 to 5 how well this song imitates {artist_dropdown.value}'s style, "
                f"where 5 is indistinguishable from the real artist.\n\n{_song_text}"
            )
            _text, _u = call_model(
                system=JUDGE_SYSTEM_PROMPT,
                user=_judge_user,
                temperature=0.0,
                schema=Judgment,
            )
            try:
                judgment = Judgment.model_validate(json.loads(_text))
                judge_output = mo.vstack(
                    [
                        mo.md(f"**Score:** {judgment.score} / 5"),
                        mo.md(f"**Reasoning:** {judgment.reasoning}"),
                        mo.callout(
                            mo.md(
                                "The judge may be biased toward its own style, longer text, or formal language. "
                                "Always calibrate LLM-as-judge scores with human annotations before using them at scale."
                            ),
                            kind="info",
                        ),
                    ]
                )
            except Exception as e:
                judge_output = mo.callout(
                    mo.md(f"Could not parse judgment: `{e}`\n\nRaw: `{_text}`"),
                    kind="danger",
                )

    judge_output
    return


if __name__ == "__main__":
    app.run()
