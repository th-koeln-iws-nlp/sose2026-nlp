import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import spacy
    import pandas as pd
    import plotly.express as px
    from collections import Counter
    from pathlib import Path

    nlp = spacy.load("en_core_web_sm")
    df = (
        pd.read_csv(Path("data/billboard_top100/billboard_top_100.csv"))
        .dropna(subset=["lyrics"])
        .reset_index(drop=True)
    )
    return Counter, df, mo, nlp, pd, px, spacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linguistic Annotations

    spaCy processes text through a **pipeline** of components. Each component adds
    annotation layers to the same `Doc` object -- from tokenization through to named
    entity recognition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        '<img src="https://spacy.io/images/pipeline.svg" '
        'style="max-width:100%;margin:1rem 0">'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll work through the three main annotation layers -- **POS tags**,
    **dependency relations**, and **named entities** -- using the Billboard Top 100
    lyrics dataset.

    ---

    ## Part 1: Part-of-Speech Tagging

    Every token in a spaCy `Doc` carries two POS annotations:

    | Attribute | Tagset | # tags | Example |
    |-----------|--------|--------|---------|
    | `token.pos_` | Universal Dependencies (UPOS) | 17 | `NOUN`, `VERB`, `ADJ` |
    | `token.tag_` | Penn Treebank | 45 | `NNS`, `VBZ`, `JJ` |

    UPOS is language-neutral and coarse; Penn Treebank is English-specific and captures
    morphological detail (singular vs. plural, verb tense, etc.).

    `spacy.explain()` translates any tag code into plain English -- works for POS tags,
    dependency labels, and NER types alike.
    """)
    return


@app.cell
def _(df, mo):
    _options = {f"{row['artist']} - {row['song_title']}": i for i, row in df.iterrows()}
    song_picker = mo.ui.dropdown(
        options=_options,
        value=list(_options.keys())[0],
        label="Pick a song",
        full_width=True,
    )
    song_picker
    return (song_picker,)


@app.cell
def _(df, nlp, song_picker):
    _text = df.loc[song_picker.value, "lyrics"]
    doc = nlp(_text)
    return (doc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Token annotation table
    """)
    return


@app.cell
def _(doc, pd, spacy):
    pd.DataFrame(
        [
            {
                "token": tok.text,
                "lemma": tok.lemma_,
                "pos (UPOS)": tok.pos_,
                "tag (PTB)": tok.tag_,
                "explain": spacy.explain(tok.tag_) or "",
                "dep": tok.dep_,
                "stop?": tok.is_stop,
            }
            for tok in doc
            if not tok.is_space
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `spacy.explain()` -- decode any tag

    `spacy.explain()` works for POS tags, Penn Treebank tags, dependency labels,
    and NER entity types -- any string that appears in a spaCy annotation:

    ```python
    spacy.explain("VBG")    # 'verb, gerund or present participle'
    spacy.explain("nsubj")  # 'nominal subject'
    spacy.explain("GPE")    # 'Countries, cities, states'
    spacy.explain("PROPN")  # 'proper noun'
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### POS distribution in Billboard lyrics

    Let's process 200 songs and see which parts of speech dominate.
    The model was trained on **news and web text** -- lyrics are a very different register.

    While we're at it, we'll also collect the most common content words (nouns and verbs,
    stopwords excluded) to see what lyrics are actually *about*.
    """)
    return


@app.cell
def _(Counter, df, nlp, pd):
    _pos_counts = Counter()
    _noun_counts = Counter()
    _verb_counts = Counter()

    for _doc in nlp.pipe(df["lyrics"].dropna().tolist(), batch_size=128):
        for _tok in _doc:
            if _tok.is_space or _tok.is_punct:
                continue
            _pos_counts[_tok.pos_] += 1
            if _tok.is_alpha and not _tok.is_stop:
                if _tok.pos_ == "NOUN":
                    _noun_counts[_tok.lemma_.lower()] += 1
                elif _tok.pos_ == "VERB":
                    _verb_counts[_tok.lemma_.lower()] += 1

    _total = sum(_pos_counts.values())
    pos_df = pd.DataFrame(
        [
            {"pos": k, "count": v, "pct": round(100 * v / _total, 1)}
            for k, v in _pos_counts.most_common()
        ]
    )
    noun_df = pd.DataFrame(
        [{"lemma": l, "count": c} for l, c in _noun_counts.most_common(15)]
    )
    verb_df = pd.DataFrame(
        [{"lemma": l, "count": c} for l, c in _verb_counts.most_common(15)]
    )
    return noun_df, pos_df, verb_df


@app.cell
def _(pos_df, px):
    px.bar(
        pos_df,
        x="pos",
        y="pct",
        text="pct",
        labels={
            "pos": "Part of Speech",
            "pct": "% of tokens (excl. punct & spaces)",
        },
        title="POS distribution across all Billboard songs",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Most common nouns (stopwords excluded)
    """)
    return


@app.cell
def _(noun_df, px):
    px.bar(
        noun_df,
        x="lemma",
        y="count",
        text="count",
        labels={"lemma": "", "count": "frequency"},
        title="Top 15 NOUN lemmas in Billboard lyrics (all songs)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Most common verbs (stopwords excluded)
    """)
    return


@app.cell
def _(px, verb_df):
    px.bar(
        verb_df,
        x="lemma",
        y="count",
        text="count",
        labels={"lemma": "", "count": "frequency"},
        title="Top 15 VERB lemmas in Billboard lyrics (all songs)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2: Dependency Parsing

    POS tags tell us *what* each word is. Dependency parsing tells us *how words relate*
    -- who is the subject, what is the object, which adjective modifies which noun.

    Each token gets a `dep_` label describing its relation to its `head`:

    | Attribute | Description |
    |-----------|-------------|
    | `token.dep_` | Relation label (`nsubj`, `obj`, `det`, `ROOT`, ...) |
    | `token.head` | Syntactic parent |
    | `token.children` | Immediate dependents |
    | `token.subtree` | All tokens below, including self |
    | `doc.noun_chunks` | Base noun phrases extracted from the tree |

    The ROOT token (usually the main verb) has no incoming arc.
    Every other token has exactly **one** head -- forming a spanning tree.

    Use `spacy.explain()` on any `dep_` label to decode it:
    ```python
    spacy.explain("nsubj")   # 'nominal subject'
    spacy.explain("advmod")  # 'adverbial modifier'
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Dependency tree -- first sentence of selected song
    """)
    return


@app.cell
def _(doc, mo, spacy):
    first_sent = next(doc.sents)
    _html = spacy.displacy.render(
        first_sent.as_doc(),
        style="dep",
        page=False,
        jupyter=False,
    )
    mo.Html(f'<div style="overflow-x:auto;padding:1rem 0">{_html}</div>')
    return (first_sent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Navigating the tree
    """)
    return


@app.cell
def _(first_sent):
    for _tok in first_sent:
        _children = [c.text for c in _tok.children]
        print(
            f"{_tok.text:15} dep={_tok.dep_:10} head={_tok.head.text:12} children={_children}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Noun chunks

    `doc.noun_chunks` extracts base noun phrases using the dependency tree.
    Each chunk exposes its **head noun** and the head's **syntactic role** in the sentence --
    useful for quickly identifying who or what sentences are about.
    """)
    return


@app.cell
def _(doc, pd):
    pd.DataFrame(
        [
            {
                "chunk": chunk.text,
                "root": chunk.root.text,
                "root.dep_": chunk.root.dep_,
                "root.head": chunk.root.head.text,
            }
            for chunk in doc.noun_chunks
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3: Named Entity Recognition

    NER finds and classifies **real-world references** in text. spaCy's English models
    (trained on OntoNotes) recognize **18 entity types** -- persons, organizations,
    locations, dates, money, and more.

    ```python
    for ent in doc.ents:
        print(ent.text, ent.label_, spacy.explain(ent.label_))
    ```

    Each entity span also exposes `token.ent_iob_` -- the **BIO tag** underlying the span:
    B = begins an entity, I = inside, O = outside.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Entity visualization -- selected song
    """)
    return


@app.cell
def _(doc, mo, spacy):
    _html = spacy.displacy.render(doc, style="ent", page=False, jupyter=False)
    mo.Html(f'<div style="line-height:2.2;padding:0.5rem">{_html}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Entity types across the dataset
    """)
    return


@app.cell
def _(Counter, df, nlp, pd, spacy):
    _ent_counts = Counter()
    for _doc in nlp.pipe(df["lyrics"].dropna().tolist(), batch_size=128):
        for _ent in _doc.ents:
            _ent_counts[_ent.label_] += 1

    ent_df = pd.DataFrame(
        [
            {"type": k, "count": v, "description": spacy.explain(k) or k}
            for k, v in _ent_counts.most_common()
        ]
    )
    ent_df
    return (ent_df,)


@app.cell
def _(ent_df, px):
    px.bar(
        ent_df,
        x="type",
        y="count",
        text="count",
        hover_data=["description"],
        labels={"type": "Entity type", "count": "Occurrences"},
        title="Named entity types in Billboard lyrics",
    )
    return


if __name__ == "__main__":
    app.run()
