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
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic

    return BERTopic, SentenceTransformer, go, mo, pd, px, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Topic Modeling with BERTopic

    BERTopic discovers topics in four modular steps: embed each lyric with a sentence-transformer, reduce dimensions (UMAP), cluster (HDBSCAN), then describe each cluster with its most distinctive words (c-TF-IDF). The number of topics is found automatically.
    """)
    return


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    docs = lyrics_df["lyrics"].fillna("").tolist()
    print(f"{len(lyrics_df)} songs loaded")
    return docs, lyrics_df


@app.cell
def _(lyrics_df):
    lyrics_df.drop(columns=["lyrics_processed"]).to_csv("data/billboard_top100/billboard_top_100.csv", index=False)

    return


@app.cell
def _(docs, lyrics_df):
    if "lyrics_processed" in lyrics_df.columns:
        processed_docs = lyrics_df["lyrics_processed"].fillna("").tolist()
        print("Using existing lyrics_processed column")
    else:
        import re
        import spacy

        nlp = spacy.load("en_core_web_sm")
        custom_stop = {
            # phonetic / nonsense
            "na",
            "choo",
            "ya",
            "mmmm",
            "ba",
            "da",
            "dee",
            "yo",
            "yi",
            "la",
            "ooh",
            "ah",
            "uh",
            "hey",
            "wanna",
            "gonna",
            "gotta",
            "ron",
            "thoia",
            # generic lyric filler that appears in almost every topic
            "know",
            "like",
            "got",
            "baby",
            "want",
            "come",
            "let",
        }
        texts = [re.sub(r"\[.*?\]", " ", doc) for doc in docs]
        processed_docs = []
        for spacy_doc in nlp.pipe(texts, batch_size=64):
            tokens = [
                token.lower_
                for token in spacy_doc
                if not token.is_stop
                and not token.is_punct
                and token.pos_ != "INTJ"
                and token.is_alpha
                and len(token) >= 3
                and token.lower_ not in custom_stop
            ]
            processed_docs.append(" ".join(tokens))
        print(f"Preprocessed {len(processed_docs)} lyrics")
    return (processed_docs,)


@app.cell
def _(lyrics_df, processed_docs):
    if "lyrics_processed" not in lyrics_df.columns:
        lyrics_df["lyrics_processed"] = processed_docs
        lyrics_df.to_csv(
            "data/billboard_top100/billboard_top_100.csv", index=False
        )
        print("Saved lyrics_processed column to CSV")
    return


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
    ## Part 1: Compute Embeddings

    Embed each lyric once with a sentence-transformer. Embeddings are reused across BERTopic runs so you can tune parameters without re-encoding.
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
        label="Embedding model",
    )
    embed_btn = mo.ui.run_button(label="Compute Embeddings")
    return embed_btn, model_dropdown


@app.cell
def _(embed_btn, mo, model_dropdown):
    mo.vstack([model_dropdown, embed_btn])
    return


@app.cell
def _(SentenceTransformer, device, docs, embed_btn, mo, model_dropdown):
    if embed_btn.value:
        mo.output.replace(
            mo.md(
                f"**Embedding {len(docs)} lyrics with `{model_dropdown.value}`...**"
            )
        )
        sbert = SentenceTransformer(model_dropdown.value, device=device)
        embeddings = sbert.encode(
            docs,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        import umap as umap_module

        reducer_2d = umap_module.UMAP(
            n_components=2, min_dist=0.0, metric="cosine", random_state=42
        )
        embeddings_2d = reducer_2d.fit_transform(embeddings)
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**{len(embeddings)} lyrics embedded.** Shape: `{embeddings.shape}`"
                ),
                kind="success",
            )
        )
    else:
        embeddings = None
        embeddings_2d = None
        mo.output.replace(
            mo.md("*Select a model and click **Compute Embeddings**.*")
        )
    return embeddings, embeddings_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Fit BERTopic

    Cluster the embeddings into topics. Adjust parameters and refit without re-embedding. `nr_topics = 0` lets HDBSCAN decide automatically.
    """)
    return


@app.cell
def _(mo):
    min_topic_size_slider = mo.ui.slider(
        start=10, stop=50, step=1, value=25, label="Min topic size"
    )
    min_df_slider = mo.ui.slider(
        start=2,
        stop=20,
        step=1,
        value=5,
        label="Min doc frequency (CountVectorizer)",
    )
    nr_topics_slider = mo.ui.slider(
        start=0, stop=30, step=1, value=0, label="Nr topics (0 = auto)"
    )
    fit_btn = mo.ui.run_button(label="Fit BERTopic")
    return fit_btn, min_df_slider, min_topic_size_slider, nr_topics_slider


@app.cell
def _(fit_btn, min_df_slider, min_topic_size_slider, mo, nr_topics_slider):
    mo.vstack([min_topic_size_slider, min_df_slider, nr_topics_slider, fit_btn])
    return


@app.cell
def _(
    BERTopic,
    embeddings,
    fit_btn,
    lyrics_df,
    min_df_slider,
    min_topic_size_slider,
    mo,
    nr_topics_slider,
    processed_docs,
):
    if fit_btn.value and embeddings is not None:
        from sklearn.feature_extraction.text import CountVectorizer

        mo.output.replace(
            mo.md("**Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)...**")
        )
        nr_topics = nr_topics_slider.value if nr_topics_slider.value > 0 else None
        vectorizer = CountVectorizer(min_df=min_df_slider.value)
        topic_model = BERTopic(
            min_topic_size=min_topic_size_slider.value,
            nr_topics=nr_topics,
            vectorizer_model=vectorizer,
            verbose=False,
        )
        topics, _probs = topic_model.fit_transform(
            processed_docs, embeddings=embeddings
        )

        mo.output.replace(mo.md("**Reducing outliers...**"))
        topics = topic_model.reduce_outliers(
            processed_docs, topics, strategy="embeddings", embeddings=embeddings
        )
        topic_model.update_topics(processed_docs, topics=topics)
        topic_info = topic_model.get_topic_info()

        n_topics = len(topic_info[topic_info["Topic"] != -1])
        n_outliers = (
            int(topic_info[topic_info["Topic"] == -1]["Count"].sum())
            if (topic_info["Topic"] == -1).any()
            else 0
        )
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**{n_topics} topics** found across {len(lyrics_df)} songs. "
                    f"{n_outliers} outliers (topic -1)."
                ),
                kind="success",
            )
        )
    elif fit_btn.value and embeddings is None:
        topic_model = None
        topics = None
        topic_info = None
        mo.output.replace(
            mo.callout(mo.md("**Compute embeddings first.**"), kind="warn")
        )
    else:
        topic_model = None
        topics = None
        topic_info = None
        mo.output.replace(mo.md("*Click **Fit BERTopic** to run topic modeling.*"))
    return topic_info, topic_model, topics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Topic Overview

    Each bar is one discovered topic, sized by document count. Topic -1 collects lyrics that did not fit any cluster (outliers).
    """)
    return


@app.cell
def _(mo, px, topic_info):
    if topic_info is None:
        sizes_fig = mo.md("*Fit BERTopic above to see topics.*")
    else:
        valid = topic_info[topic_info["Topic"] != -1].sort_values(
            "Count", ascending=False
        )
        sizes_fig = px.bar(
            valid,
            x="Name",
            y="Count",
            title="Topic Sizes",
            labels={"Name": "Topic", "Count": "Documents"},
        )
        sizes_fig.update_layout(xaxis_tickangle=-45)
    sizes_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4: Topic Explorer

    Select a topic to inspect its top words and their c-TF-IDF scores — how much more frequent a word is in this cluster compared to all others.
    """)
    return


@app.cell
def _(mo, topic_info):
    if topic_info is not None:
        valid_topics = topic_info[topic_info["Topic"] != -1]
        topic_options = {
            row["Name"]: row["Topic"] for _, row in valid_topics.iterrows()
        }
        topic_selector = mo.ui.dropdown(options=topic_options, label="Topic")
    else:
        topic_selector = mo.ui.dropdown(
            options={"(fit BERTopic first)": None}, label="Topic"
        )
    topic_selector
    return (topic_selector,)


@app.cell
def _(mo, px, topic_info, topic_model, topic_selector):
    if topic_model is None or topic_selector.value is None:
        words_fig = mo.md("*Fit BERTopic and select a topic above.*")
    else:
        words_scores = topic_model.get_topic(topic_selector.value)
        words = [w for w, _ in words_scores]
        scores = [s for _, s in words_scores]
        topic_name = topic_info[topic_info["Topic"] == topic_selector.value][
            "Name"
        ].iloc[0]
        words_fig = px.bar(
            x=scores,
            y=words,
            orientation="h",
            title=f"Top Words: {topic_name}",
            labels={"x": "c-TF-IDF Score", "y": ""},
        )
        words_fig.update_layout(yaxis={"autorange": "reversed"})
    words_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5: Document Map

    Each point is one song projected to 2D via UMAP and colored by topic. Hover for artist and title. Grey points are outliers.
    """)
    return


@app.cell
def _(embeddings_2d, go, lyrics_df, mo, topic_info, topic_model, topics):
    if embeddings_2d is None or topic_model is None:
        doc_map = mo.md("*Fit BERTopic above to see the document map.*")
    else:
        map_df = lyrics_df[["artist", "song_title", "year"]].copy()
        map_df["x"] = embeddings_2d[:, 0]
        map_df["y"] = embeddings_2d[:, 1]
        map_df["topic"] = topics

        topic_name_map = {
            row["Topic"]: row["Name"] for _, row in topic_info.iterrows()
        }
        map_df["topic_name"] = (
            map_df["topic"].map(topic_name_map).fillna("Outlier")
        )

        traces = []

        outliers = map_df[map_df["topic"] == -1]
        if len(outliers):
            traces.append(
                go.Scatter(
                    x=outliers["x"],
                    y=outliers["y"],
                    mode="markers",
                    marker={"color": "lightgrey", "size": 4, "opacity": 0.4},
                    text=outliers.apply(
                        lambda r: f"{r['artist']} - {r['song_title']}", axis=1
                    ),
                    hovertemplate="%{text}<extra>Outlier</extra>",
                    name="Outlier",
                )
            )

        for _tid in sorted(map_df[map_df["topic"] != -1]["topic"].unique()):
            _subset = map_df[map_df["topic"] == _tid]
            _name = topic_name_map.get(_tid, str(_tid))
            traces.append(
                go.Scatter(
                    x=_subset["x"],
                    y=_subset["y"],
                    mode="markers",
                    marker={"size": 7, "opacity": 0.85},
                    text=_subset.apply(
                        lambda r: f"{r['artist']} - {r['song_title']}", axis=1
                    ),
                    hovertemplate="%{text}<extra>" + _name + "</extra>",
                    name=_name,
                )
            )

        doc_map = go.Figure(traces)
        doc_map.update_layout(
            title="Document Map: Songs by Topic",
            xaxis={"title": "UMAP Dim 1", "showgrid": False},
            yaxis={"title": "UMAP Dim 2", "showgrid": False},
        )
    doc_map
    return


if __name__ == "__main__":
    app.run()
