import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    return (
        CountVectorizer,
        TfidfVectorizer,
        cosine_similarity,
        go,
        mo,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vectorization Tutorial: Bag of Words and TF-IDF

    This tutorial applies two fundamental text vectorization methods to Billboard Top 100 song lyrics.

    You will learn to:
    - Convert text to numeric vectors with `CountVectorizer` and `TfidfVectorizer`
    - Understand sparse matrix representation
    - See how TF-IDF improves on raw word counts
    - Find similar songs using cosine similarity
    """)
    return


@app.cell
def _(pd):
    lyrics_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    print(f"{len(lyrics_df)} songs")
    return (lyrics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Bag of Words

    The core idea: assign each unique word in the corpus an index, then represent each document as a vector of word counts.

    `fit_transform(corpus)` does two things in one call:
    1. **fit**: scan all documents and build the vocabulary
    2. **transform**: convert each document into a count vector

    The result is a **document-term matrix**: rows are songs, columns are vocabulary terms.
    """)
    return


@app.cell
def _(CountVectorizer, lyrics_df):
    cv_corpus = CountVectorizer(min_df=2)
    X_corpus = cv_corpus.fit_transform(lyrics_df["lyrics"])
    n_cells = X_corpus.shape[0] * X_corpus.shape[1]
    sparsity = (1 - X_corpus.nnz / n_cells) * 100
    print(f"Shape:            {X_corpus.shape[0]:,} songs x {X_corpus.shape[1]:,} terms")
    print(f"Non-zero entries: {X_corpus.nnz:,} out of {n_cells:,} total")
    print(f"Sparsity:         {sparsity:.1f}% of entries are zero")
    return X_corpus, cv_corpus


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sparsity

    Over 99% of the matrix entries are zero. Most songs use only a small fraction of the full vocabulary. Scikit-learn stores only the non-zero entries using `scipy.sparse`, which is far more memory-efficient than a full dense array.

    > Never call `.toarray()` on a large corpus matrix. It would materialize all those zeros in memory.

    ### Most Frequent Terms
    """)
    return


@app.cell
def _(X_corpus, cv_corpus, pd, px):
    term_sums = X_corpus.sum(axis=0).A1
    freq_df = pd.DataFrame(
        {"term": cv_corpus.get_feature_names_out(), "count": term_sums}
    ).sort_values("count", ascending=False).head(30)

    fig_freq = px.bar(
        freq_df,
        x="term",
        y="count",
        title="Top 30 Most Frequent Terms (no stop word filter)",
        labels={"term": "Term", "count": "Total Count"},
    )
    fig_freq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generic words dominate the counts. This is precisely the problem TF-IDF was designed to solve.

    ### Vocabulary Control

    `CountVectorizer` has several parameters to filter what goes into the vocabulary:

    | Parameter | What it does |
    |-----------|-------------|
    | `stop_words='english'` | Remove very common English words |
    | `min_df` | Ignore terms appearing in fewer than N documents (removes typos and ultra-rare words) |
    | `max_features` | Keep only the top N most frequent terms (caps vocabulary size) |

    Adjust the controls below to see how filtering changes the top terms.
    """)
    return


@app.cell
def _(mo):
    max_features_slider = mo.ui.slider(500, 10000, step=500, value=3000, label="max_features")
    min_df_slider = mo.ui.slider(1, 20, step=1, value=3, label="min_df")
    stop_words_toggle = mo.ui.checkbox(value=True, label="Remove English stop words")
    mo.vstack([max_features_slider, min_df_slider, stop_words_toggle])
    return max_features_slider, min_df_slider, stop_words_toggle


@app.cell
def _(
    CountVectorizer,
    lyrics_df,
    max_features_slider,
    min_df_slider,
    pd,
    px,
    stop_words_toggle,
):
    cv_interactive = CountVectorizer(
        max_features=max_features_slider.value,
        min_df=min_df_slider.value,
        stop_words="english" if stop_words_toggle.value else None,
    )
    X_interactive = cv_interactive.fit_transform(lyrics_df["lyrics"])
    interactive_sums = X_interactive.sum(axis=0).A1

    interactive_freq_df = pd.DataFrame(
        {"term": cv_interactive.get_feature_names_out(), "count": interactive_sums}
    ).sort_values("count", ascending=False).head(30)

    sw_label = "english" if stop_words_toggle.value else "None"
    fig_interactive = px.bar(
        interactive_freq_df,
        x="term",
        y="count",
        title=f"Top 30 Terms (max_features={max_features_slider.value}, min_df={min_df_slider.value}, stop_words={sw_label})",
        labels={"term": "Term", "count": "Total Count"},
    )
    fig_interactive
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### N-grams

    Each feature so far is a single word (**unigram**). Sometimes word combinations carry meaning that individual words miss. The classic example: "not good" as a **bigram** captures negation, while "not" and "good" as separate unigrams do not.

    In sklearn: `ngram_range=(1, 2)` includes both unigrams and bigrams.
    """)
    return


@app.cell
def _(CountVectorizer, lyrics_df, pd, px):
    ngram_configs = [
        ("unigrams (1,1)", (1, 1)),
        ("+ bigrams (1,2)", (1, 2)),
        ("+ trigrams (1,3)", (1, 3)),
    ]
    ngram_results = []
    for label, ngram_range in ngram_configs:
        cv_ngram = CountVectorizer(ngram_range=ngram_range, min_df=2, stop_words="english")
        cv_ngram.fit(lyrics_df["lyrics"])
        ngram_results.append({"n-gram setting": label, "vocabulary size": len(cv_ngram.vocabulary_)})

    ngram_df = pd.DataFrame(ngram_results)
    print(ngram_df.to_string(index=False))

    fig_ngram = px.bar(
        ngram_df,
        x="n-gram setting",
        y="vocabulary size",
        title="Vocabulary Size by N-gram Setting (min_df=2, stop_words='english')",
        labels={"n-gram setting": "N-gram Range", "vocabulary size": "Vocabulary Size"},
        text="vocabulary size",
    )
    fig_ngram.update_traces(textposition="outside")
    fig_ngram
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adding bigrams or trigrams multiplies vocabulary size substantially. A corpus with 5,000 unigrams might generate hundreds of thousands of bigrams. Use `max_features` to keep it manageable.

    ---

    ## Part 2: TF-IDF

    ### The Problem with Raw Counts

    Words like "the", "you", or "love" appear in almost every song and dominate raw counts, but they carry little information about what makes any particular song distinctive.

    **TF-IDF** (Term Frequency, Inverse Document Frequency) was introduced by Karen Sparck Jones in 1972. It combines two signals:

    | Component | Measures | Effect |
    |-----------|----------|--------|
    | **TF** | How often the word appears in *this* document | Rewards locally frequent words |
    | **IDF** | How rare the word is *across all* documents | Downweights ubiquitous words, upweights distinctive ones |
    | **TF-IDF** | TF x IDF | High for words frequent here but rare across the corpus |

    ### TF-IDF on the Full Corpus
    """)
    return


@app.cell
def _(TfidfVectorizer, lyrics_df):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=2, stop_words="english")
    X_tfidf = tfidf_vectorizer.fit_transform(lyrics_df["lyrics"])
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    return X_tfidf, tfidf_vectorizer


@app.cell
def _(CountVectorizer, lyrics_df):
    cv_lyrics = CountVectorizer(max_features=5000, min_df=2, stop_words="english")
    X_bow = cv_lyrics.fit_transform(lyrics_df["lyrics"])
    return X_bow, cv_lyrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top TF-IDF Terms per Song

    Select a song to see which words TF-IDF considers most distinctive compared to the rest of the corpus.
    """)
    return


@app.cell
def _(lyrics_df, mo):
    song_options = [
        f"{artist} - {title} ({int(year)})"
        for artist, title, year in zip(lyrics_df["artist"], lyrics_df["song_title"], lyrics_df["year"])
    ]
    song_dropdown = mo.ui.dropdown(
        options=song_options,
        value=song_options[0],
        label="Select a song",
    )
    song_dropdown
    return song_dropdown, song_options


@app.cell
def _(lyrics_df, song_dropdown, song_options):
    selected_idx = song_options.index(song_dropdown.value)
    selected_song = lyrics_df.iloc[selected_idx]
    return selected_idx, selected_song


@app.cell
def _(X_tfidf, pd, px, selected_idx, selected_song, tfidf_vectorizer):
    tfidf_row = X_tfidf[selected_idx].toarray().flatten()
    top_tfidf_df = pd.DataFrame(
        {"term": tfidf_vectorizer.get_feature_names_out(), "tfidf": tfidf_row}
    ).sort_values("tfidf", ascending=False).head(20)

    fig_tfidf_song = px.bar(
        top_tfidf_df,
        x="tfidf",
        y="term",
        orientation="h",
        title=f"Top 20 TF-IDF Terms: {selected_song['artist']} - {selected_song['song_title']}",
        labels={"tfidf": "TF-IDF Score", "term": ""},
    )
    fig_tfidf_song.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_tfidf_song
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BoW vs. TF-IDF: Side-by-Side

    Let's compare raw word counts and TF-IDF scores for the same song. BoW values are normalized to the 0-1 range for visual comparison.
    """)
    return


@app.cell
def _(
    X_bow,
    X_tfidf,
    cv_lyrics,
    go,
    np,
    selected_idx,
    selected_song,
    tfidf_vectorizer,
):
    bow_vec = X_bow[selected_idx].toarray().flatten()
    tfidf_vec = X_tfidf[selected_idx].toarray().flatten()

    top20_idx = tfidf_vec.argsort()[::-1][:20]
    top20_terms = tfidf_vectorizer.get_feature_names_out()[top20_idx]
    top20_tfidf_scores = tfidf_vec[top20_idx]

    bow_vocab_lookup = {t: i for i, t in enumerate(cv_lyrics.get_feature_names_out())}
    top20_bow = np.array(
        [bow_vec[bow_vocab_lookup[t]] if t in bow_vocab_lookup else 0.0 for t in top20_terms],
        dtype=float,
    )
    bow_max = top20_bow.max()
    top20_bow_norm = top20_bow / bow_max if bow_max > 0 else top20_bow

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(name="BoW (normalized)", x=top20_terms, y=top20_bow_norm))
    fig_compare.add_trace(go.Bar(name="TF-IDF", x=top20_terms, y=top20_tfidf_scores))
    fig_compare.update_layout(
        barmode="group",
        title=f"BoW vs. TF-IDF: {selected_song['artist']} - {selected_song['song_title']}",
        xaxis_title="Term",
        yaxis_title="Score",
        legend={"orientation": "h"},
    )
    fig_compare
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## PyMDE Projection of TF-IDF Vectors

    TF-IDF vectors have thousands of dimensions, one per vocabulary term. **PyMDE** (Minimum Distortion Embedding) reduces them to 2D while preserving local neighborhood structure, so songs with similar lyrics land near each other.

    Coloring by genre reveals whether lyrically similar songs also share genre labels. Songs without a Spotify genre label appear as "Unknown" (~42% of the corpus).
    """)
    return


@app.cell
def _(X_tfidf, lyrics_df):
    import pymde

    mde = pymde.preserve_neighbors(X_tfidf, embedding_dim=2, verbose=False)
    mde_embedding = mde.embed(verbose=False).numpy()

    umap_df = lyrics_df[["artist", "song_title", "year", "genre"]].copy()
    umap_df["x"] = mde_embedding[:, 0]
    umap_df["y"] = mde_embedding[:, 1]
    umap_df["genre_label"] = umap_df["genre"].fillna("Unknown")
    return (umap_df,)


@app.cell
def _(px, umap_df):
    fig_umap = px.scatter(
        umap_df,
        x="x",
        y="y",
        color="genre_label",
        hover_data={"artist": True, "song_title": True, "year": True, "x": False, "y": False},
        title="PyMDE Projection of TF-IDF Vectors, Colored by Genre",
        labels={"genre_label": "Genre", "x": "PyMDE 1", "y": "PyMDE 2"},
        opacity=0.7,
    )
    fig_umap.update_traces(marker={"size": 5})
    fig_umap.update_layout(legend_title_text="Genre")
    fig_umap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application: Song Similarity Search

    TF-IDF vectors capture each song's "topic signature". We can find similar songs by computing **cosine similarity** between their TF-IDF vectors. A score of 1.0 means identical direction, 0.0 means no overlap at all.

    Enter lyrics, keywords, or a short phrase to find the most lyrically similar songs in the corpus.
    """)
    return


@app.cell
def _(mo):
    query_input = mo.ui.text(
        placeholder="e.g. heartbreak love forever",
        label="Search query",
    )
    query_input
    return (query_input,)


@app.cell
def _(
    X_tfidf,
    cosine_similarity,
    lyrics_df,
    mo,
    query_input,
    tfidf_vectorizer,
):
    if query_input.value.strip():
        query_vec = tfidf_vectorizer.transform([query_input.value])
        sim_scores = cosine_similarity(query_vec, X_tfidf).flatten()
        top_sim_idx = sim_scores.argsort()[::-1][:10]
        result_df = lyrics_df.iloc[top_sim_idx][["artist", "song_title", "year", "genre", "lyrics"]].copy()
        result_df["similarity"] = sim_scores[top_sim_idx].round(3)
        result_df = result_df.reset_index(drop=True)
        search_output = result_df
    else:
        sim_scores = None
        top_sim_idx = None
        search_output = mo.md("*Enter a query above to search for similar songs.*")
    search_output
    return sim_scores, top_sim_idx


@app.cell
def _(go, mo, sim_scores, top_sim_idx, umap_df):
    if top_sim_idx is None:
        umap_search_out = mo.md("*Results will appear on the map once you enter a query.*")
    else:
        highlight = umap_df.iloc[top_sim_idx].copy()
        highlight["similarity"] = sim_scores[top_sim_idx].round(3)

        fig_umap_sim = go.Figure()
        fig_umap_sim.add_trace(go.Scatter(
            x=umap_df["x"],
            y=umap_df["y"],
            mode="markers",
            marker={"color": "lightgrey", "size": 3, "opacity": 0.4},
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_umap_sim.add_trace(go.Scatter(
            x=highlight["x"],
            y=highlight["y"],
            mode="markers",
            marker={"color": "crimson", "size": 11, "opacity": 0.9,
                    "line": {"width": 1, "color": "white"}},
            text=highlight.apply(
                lambda r: f"{r['artist']} - {r['song_title']} ({int(r['year'])})<br>similarity: {r['similarity']}",
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
            name="Top 10 matches",
        ))
        fig_umap_sim.update_layout(
            title="Similar Songs in TF-IDF Space",
            xaxis_title="PyMDE 1",
            yaxis_title="PyMDE 2",
            xaxis={"showgrid": False},
            yaxis={"showgrid": False},
        )
        umap_search_out = fig_umap_sim
    umap_search_out
    return


if __name__ == "__main__":
    app.run()
