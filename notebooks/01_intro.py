import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import io
    import marimo as mo
    import pandas as pd
    import spacy
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    from wordfreq import zipf_frequency

    return WordCloud, go, io, mo, pd, plt, re, spacy, zipf_frequency


@app.cell
def _(pd):
    df = pd.read_csv("data/billboard_top100/all_songs_data_processed.csv")
    artists = sorted(df["Artist"].dropna().unique().tolist())
    return artists, df


@app.cell
def _(spacy):
    nlp = spacy.load("en_core_web_sm")
    return (nlp,)


@app.cell
def _(mo):
    mo.md("""
    ## Word Cloud
    """)
    return


@app.cell
def _(df, nlp):
    _MANUAL_FILLERS = {
        "ooh",
        "ohh",
        "ahh",
        "mmm",
        "hmm",
        "hm",
        "uh",
        "um",
        "na",
        "la",
        "da",
        "ba",
        "ya",
        "yo",
        "woah",
        "whoa",
        "gonna",
        "wanna",
        "gotta",
        "ain",
        "til",
        "em",
        "oo",
    }
    _vocab = set(" ".join(df["Corpus"].dropna()).lower().split())
    _docs = list(nlp.pipe(sorted(_vocab), batch_size=500))
    _interjections = {doc[0].text for doc in _docs if doc[0].pos_ == "INTJ"}
    lyric_stopwords = _interjections | _MANUAL_FILLERS
    noun_vocab = {doc[0].text for doc in _docs if doc[0].pos_ == "NOUN"}
    return lyric_stopwords, noun_vocab


@app.cell
def _(artists, mo):
    artist_dropdown = mo.ui.dropdown(
        options=artists,
        value=artists[0],
        label="Artist",
    )
    artist_dropdown
    return (artist_dropdown,)


@app.cell
def _(
    WordCloud,
    artist_dropdown,
    df,
    io,
    lyric_stopwords,
    mo,
    noun_vocab,
    plt,
):
    _artist_songs = df[df["Artist"] == artist_dropdown.value]
    _corpus = " ".join(
        w
        for text in _artist_songs["Corpus"].dropna()
        for w in text.split()
        if w.lower() in noun_vocab
    )

    _wc = WordCloud(
        width=900,
        height=420,
        background_color="white",
        max_words=10,
        colormap="viridis",
        collocations=False,
        stopwords=lyric_stopwords,
    ).generate(_corpus)

    _fig, _ax = plt.subplots(figsize=(11, 5))
    _ax.imshow(_wc, interpolation="bilinear")
    _ax.axis("off")
    _ax.set_title(
        f"{artist_dropdown.value}  ·  {len(_artist_songs)} song(s) in dataset",
        fontsize=13,
        pad=10,
    )
    plt.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=130)
    plt.close(_fig)
    _buf.seek(0)
    mo.image(_buf.read(), width="100%")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Lyrics Statistics Over Time
    """)
    return


@app.cell
def _(df):
    _df = df.copy()
    _df["lexical_diversity"] = _df["Unique Word Counts"] / _df["Word Counts"]
    _yearly = (
        _df.groupby("Year")
        .agg(
            songs=("Song Title", "count"),
            avg_word_count=("Word Counts", "mean"),
            avg_unique_words=("Unique Word Counts", "mean"),
            avg_lexical_diversity=("lexical_diversity", "mean"),
        )
        .reset_index()
    )
    yearly_stats = _yearly[_yearly["songs"] >= 5].copy()
    return (yearly_stats,)


@app.cell
def _(mo):
    _metric_options = {
        "Songs in dataset": "songs",
        "Avg. word count": "avg_word_count",
        "Avg. unique words": "avg_unique_words",
        "Lexical diversity (unique/total)": "avg_lexical_diversity",
    }
    metric_dropdown = mo.ui.dropdown(
        options=list(_metric_options.keys()),
        value="Avg. word count",
        label="Metric",
    )
    metric_col = _metric_options
    metric_dropdown
    return metric_col, metric_dropdown


@app.cell
def _(go, metric_col, metric_dropdown, yearly_stats):
    _label = metric_dropdown.value
    _col = metric_col[_label]
    _smoothed = yearly_stats.set_index("Year")[_col].rolling(5, center=True).mean()

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=yearly_stats["Year"],
            y=yearly_stats[_col],
            mode="markers",
            name="per year",
            marker=dict(color="#a0c4e8", size=5, opacity=0.6),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_smoothed.index.tolist(),
            y=_smoothed.values.tolist(),
            mode="lines",
            name="5-yr avg",
            line=dict(color="#1f6fab", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31,111,171,0.08)",
        )
    )
    _fig.update_layout(
        title=f"{_label} · Billboard Hot 100 (1959–2023)",
        xaxis_title="Year",
        yaxis_title=_label,
        legend=dict(orientation="h", y=1.08),
        height=380,
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Vocabulary Rankings
    """)
    return


@app.cell
def _(df, pd, zipf_frequency):
    # Only artists with ≥5 songs for reliable stats
    _MIN_SONGS = 5

    def _artist_stats(group):
        words = " ".join(group["Corpus"].dropna()).lower().split()
        unique = set(words)
        zipf_scores = [zipf_frequency(w, "en") for w in unique]
        avg_zipf = sum(zipf_scores) / len(zipf_scores) if zipf_scores else 0
        return pd.Series(
            {
                "n_songs": len(group),
                "total_unique_words": len(unique),
                "avg_unique_per_song": group["Unique Word Counts"].mean(),
                "avg_zipf": avg_zipf,
            }
        )

    artist_stats = (
        df.groupby("Artist").apply(_artist_stats, include_groups=False).reset_index()
    )
    artist_stats = artist_stats[artist_stats["n_songs"] >= _MIN_SONGS]
    return (artist_stats,)


@app.cell
def _(artist_stats, go, mo):
    _n = 10
    _top = artist_stats.nlargest(_n, "avg_unique_per_song")
    _bot = artist_stats.nsmallest(_n, "avg_unique_per_song")

    def _bar(data, col, title, color):
        _sorted = data.sort_values(col)  # ascending → largest at top
        return go.Figure(
            go.Bar(
                x=_sorted[col].round(1),
                y=_sorted["Artist"],
                orientation="h",
                marker_color=color,
                hovertemplate="%{y}: %{x:.1f} unique words/song<extra></extra>",
            )
        ).update_layout(
            title=title,
            height=80 + _n * 28,
            margin=dict(l=10, r=20, t=50, b=30),
            xaxis_title="Avg. unique words per song",
        )

    mo.hstack(
        [
            _bar(
                _top, "avg_unique_per_song", f"Top {_n} — Largest Vocabulary", "#2a9d8f"
            ),
            _bar(
                _bot,
                "avg_unique_per_song",
                f"Bottom {_n} — Smallest Vocabulary",
                "#e76f51",
            ),
        ],
        widths=[1, 1],
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Word Rarity

    Each word is scored using its **Zipf frequency** in English
    (scale 1–7: 7 = very common like *the*, 1 = very rare).
    A lower average score means the artist tends to use rarer, less common words.
    """)
    return


@app.cell
def _(artist_stats, go, mo):
    _n = 10
    _rarest = artist_stats.nsmallest(_n, "avg_zipf")
    _commonest = artist_stats.nlargest(_n, "avg_zipf")

    def _rarity_bar(data, title, color):
        _sorted = data.sort_values(
            "avg_zipf", ascending=False
        )  # rarest (lowest) ends up on top
        return go.Figure(
            go.Bar(
                x=_sorted["avg_zipf"].round(3),
                y=_sorted["Artist"],
                orientation="h",
                marker_color=color,
                hovertemplate="%{y}: Zipf avg %{x:.3f}<extra></extra>",
            )
        ).update_layout(
            title=title,
            height=80 + _n * 28,
            margin=dict(l=10, r=20, t=50, b=30),
            xaxis_title="Avg. Zipf frequency (lower = rarer)",
        )

    mo.hstack(
        [
            _rarity_bar(_rarest, f"Top {_n} — Rarest Vocabulary", "#6a4c93"),
            _rarity_bar(_commonest, f"Top {_n} — Most Common Vocabulary", "#f4a261"),
        ],
        widths=[1, 1],
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Find Songs About
    """)
    return


@app.cell
def _(mo):
    search_input = mo.ui.text(
        placeholder="e.g. love, rain, dance …",
        label="Search lyrics for",
    )
    search_input
    return (search_input,)


@app.cell
def _(df, mo, nlp, re, search_input):
    _query = search_input.value.strip()

    if not _query:
        _result = mo.md("_Type a word above to search._")
    else:
        _lemma = nlp(_query.lower())[0].lemma_
        _pattern = re.compile(rf"\b{re.escape(_lemma)}\b", re.IGNORECASE)
        _matches = df[df["Corpus"].fillna("").apply(lambda c: bool(_pattern.search(c)))]

        if _matches.empty:
            _result = mo.md(f"No songs found for **{_query}** (lemma: *{_lemma}*).")
        else:

            def _highlight_lyrics(lyrics: str) -> str:
                if not isinstance(lyrics, str):
                    return ""
                doc = nlp(lyrics)
                out = []
                last = 0
                for token in doc:
                    if token.lemma_.lower() == _lemma.lower():
                        out.append(lyrics[last : token.idx])
                        out.append(
                            f'<mark style="background:#ffe066;border-radius:3px;padding:0 2px">'
                            f"{token.text}</mark>"
                        )
                        last = token.idx + len(token.text)
                out.append(lyrics[last:])
                return "".join(out)

            _cards = []
            for _, _row in _matches.iterrows():
                _highlighted = _highlight_lyrics(_row["Lyrics"])
                _year = int(_row["Year"]) if _row["Year"] == _row["Year"] else "?"
                _cards.append(
                    mo.callout(
                        mo.Html(
                            f"<strong>{_row['Song Title']}</strong>"
                            f"&nbsp;·&nbsp;<span style='color:#666'>{_row['Artist']}</span>"
                            f"&nbsp;·&nbsp;<span style='color:#999;font-size:0.85em'>{_year}</span>"
                            "<hr style='margin:6px 0;border-color:#eee'>"
                            "<div style='font-size:0.88em;line-height:1.7;"
                            "max-height:200px;overflow-y:auto;white-space:pre-wrap'>"
                            f"{_highlighted}</div>"
                        ),
                        kind="info",
                    )
                )

            _result = mo.vstack(
                [
                    mo.md(
                        f"**{len(_matches)} song(s)** matching *{_query}* (lemma: `{_lemma}`)"
                    ),
                    mo.vstack(_cards),
                ]
            )

    _result
    return


if __name__ == "__main__":
    app.run()
