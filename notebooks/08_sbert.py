import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import torch
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer, cosine_similarity, go, mo, pd, px, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semantic Search with Sentence Transformers

    Sentence embeddings with [`sentence-transformers`](https://www.sbert.net/) applied to Billboard Top 100 lyrics: embed the corpus once, then search by meaning.
    """)
    return


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    print(f"{len(lyrics_df)} songs loaded")
    return (lyrics_df,)


@app.cell
def _(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_label = "MPS (Apple Silicon Chip)"
    else:
        device = torch.device("cpu")
        device_label = "CPU"
    mo.callout(mo.md(f"Device: **{device_label}**"), kind="info")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Embedding the Corpus

    `SentenceTransformer.encode()` returns a numpy array of shape `(n_songs, embedding_dim)`.
    Each row is a fixed-size dense vector representing the meaning of one lyric.
    All three models below are bi-encoders: each text is encoded independently, so the full corpus can be pre-computed once.
    """)
    return


@app.cell
def _(mo):
    model_dropdown = mo.ui.dropdown(
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ],
        value="all-MiniLM-L6-v2",
        label="Model",
    )
    embed_btn = mo.ui.run_button(label="Compute Embeddings")
    return embed_btn, model_dropdown


@app.cell
def _(embed_btn, mo, model_dropdown):
    mo.vstack([model_dropdown, embed_btn])
    return


@app.cell
def _(SentenceTransformer, device, embed_btn, lyrics_df, mo, model_dropdown):
    if embed_btn.value:
        sbert_model = SentenceTransformer(model_dropdown.value, device=device)
        mo.output.replace(
            mo.md(
                f"**Encoding {len(lyrics_df)} lyrics with `{model_dropdown.value}`...**"
            )
        )
        corpus_embeddings = sbert_model.encode(
            lyrics_df["lyrics"].fillna("").tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**{len(corpus_embeddings)} songs encoded** with `{model_dropdown.value}`.  "
                    f"Embedding shape: `{corpus_embeddings.shape}`"
                ),
                kind="success",
            )
        )
    else:
        corpus_embeddings = None
        mo.output.replace(
            mo.md("*Select a model and click **Compute Embeddings**.*")
        )
    return (corpus_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: 2D Projection

    PyMDE reduces the high-dimensional embeddings to 2D. Songs close in meaning land near each other, regardless of shared vocabulary.
    """)
    return


@app.cell
def _(corpus_embeddings, device, lyrics_df):
    if corpus_embeddings is None:
        proj_df = None
    else:
        import pymde

        mde = pymde.preserve_neighbors(
            corpus_embeddings, embedding_dim=2, verbose=False, device=device
        )
        emb_2d = mde.embed(verbose=False).cpu().numpy()

        proj_df = lyrics_df[["artist", "song_title", "year", "genre"]].copy()
        proj_df["x"] = emb_2d[:, 0]
        proj_df["y"] = emb_2d[:, 1]
        proj_df["genre_label"] = proj_df["genre"].fillna("Unknown")
    return (proj_df,)


@app.cell
def _(mo, model_dropdown, proj_df, px):
    if proj_df is None:
        proj_fig = mo.md("*Compute embeddings above to see the 2D projection.*")
    else:
        proj_fig = px.scatter(
            proj_df,
            x="x",
            y="y",
            color="genre_label",
            hover_data={
                "artist": True,
                "song_title": True,
                "year": True,
                "x": False,
                "y": False,
            },
            title=f"PyMDE Projection of SBERT Embeddings ({model_dropdown.value}), Colored by Genre",
            labels={"genre_label": "Genre", "x": "Dim 1", "y": "Dim 2"},
            opacity=0.7,
        )
        proj_fig.update_traces(marker={"size": 5})
        proj_fig.update_layout(legend_title_text="Genre")
    proj_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Semantic Search

    The query is embedded with the same model, then ranked by cosine similarity against the stored corpus vectors.
    Try queries that share no words with the expected results, e.g. *"heartbreak and moving on"* or *"party and dancing all night"*.
    """)
    return


@app.cell
def _(mo):
    query_input = mo.ui.text(
        placeholder="e.g. heartbreak and moving on",
        label="Search query",
    )
    query_input
    return (query_input,)


@app.cell
def _(
    SentenceTransformer,
    corpus_embeddings,
    cosine_similarity,
    device,
    lyrics_df,
    mo,
    model_dropdown,
    query_input,
):
    if corpus_embeddings is None:
        search_state = None
        search_out = mo.callout(
            mo.md("**Compute embeddings first.**"), kind="warn"
        )
    elif not query_input.value.strip():
        search_state = None
        search_out = mo.md(
            "*Enter a query above to search for lyrically similar songs by meaning.*"
        )
    else:
        search_model = SentenceTransformer(model_dropdown.value, device=device)
        query_emb = search_model.encode(
            [query_input.value.strip()], convert_to_numpy=True
        )
        scores = cosine_similarity(query_emb, corpus_embeddings).flatten()
        top_idx = scores.argsort()[::-1][:10]

        result_df = lyrics_df.iloc[top_idx][
            ["artist", "song_title", "year", "genre", "lyrics"]
        ].copy()
        result_df["similarity"] = scores[top_idx].round(3)
        result_df = result_df.reset_index(drop=True)

        search_state = {"top_idx": top_idx, "scores": scores}
        search_out = result_df
    search_out
    return (search_state,)


@app.cell
def _(go, mo, proj_df, search_state):
    if proj_df is None or search_state is None:
        map_out = mo.md(
            "*Search results will appear on the map once embeddings are computed and a query is entered.*"
        )
    else:
        match_idx = search_state["top_idx"]
        highlight = proj_df.iloc[match_idx].copy()
        highlight["similarity"] = search_state["scores"][match_idx].round(3)

        search_map = go.Figure()
        search_map.add_trace(
            go.Scatter(
                x=proj_df["x"],
                y=proj_df["y"],
                mode="markers",
                marker={"color": "lightgrey", "size": 3, "opacity": 0.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        search_map.add_trace(
            go.Scatter(
                x=highlight["x"],
                y=highlight["y"],
                mode="markers",
                marker={
                    "color": "crimson",
                    "size": 11,
                    "opacity": 0.9,
                    "line": {"width": 1, "color": "white"},
                },
                text=highlight.apply(
                    lambda r: (
                        f"{r['artist']} - {r['song_title']} ({int(r['year'])})<br>"
                        f"similarity: {r['similarity']}"
                    ),
                    axis=1,
                ),
                hovertemplate="%{text}<extra></extra>",
                name="Top 10 matches",
            )
        )
        search_map.update_layout(
            title="Semantic Search Results in Embedding Space",
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            xaxis={"showgrid": False},
            yaxis={"showgrid": False},
        )
        map_out = search_map
    map_out
    return


if __name__ == "__main__":
    app.run()
