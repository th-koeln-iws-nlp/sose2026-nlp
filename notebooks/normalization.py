import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import spacy
    from nltk.stem import PorterStemmer

    nlp = spacy.load("en_core_web_sm")
    stemmer = PorterStemmer()
    return mo, nlp, stemmer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Text Normalization

    Normalization means putting tokens into a **standard format** — choosing a single
    normal form for words that appear in multiple forms. This reduces vocabulary size
    and helps downstream tasks treat equivalent words the same way.

    We continue with our running example:
    """)
    return


@app.cell
def _():
    TEXT = "And Dr. Dre said… nothing, you idiots! Dr. Dre's dead, he's locked in my basement (ha ha!)"
    print(TEXT)
    return (TEXT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · Case folding

    The simplest normalization: convert everything to lowercase.
    """)
    return


@app.cell
def _(TEXT):
    lowered = TEXT.lower()
    print(lowered)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `"And"`, `"and"`, `"AND"` all become `"and"` — this immediately reduces vocabulary size.

    **But be careful:** case carries information. `"US"` (country) vs. `"us"` (pronoun),
    `"Apple"` (company) vs. `"apple"` (fruit). For lyrics analysis, lowercasing is almost
    always fine. For other tasks, think before you fold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · Stemming

    Stemming reduces words to their base by **stripping suffixes** using heuristic rules.
    It's fast and needs no dictionary — but can produce non-words.
    """)
    return


@app.cell
def _(stemmer):
    words = [
        "locked",
        "locking",
        "locks",
        "lock",
        "studies",
        "studying",
        "studied",
        "ran",
        "running",
        "runner",
    ]

    for w in words:
        print(f"{w:>12} → {stemmer.stem(w)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Two problems are visible:

    - Produces **non-words** (`"studi"` instead of `"study"`)
    - Doesn't connect **irregular forms** (`"ran"` stays `"ran"`, not `"run"`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's stem our running example:
    """)
    return


@app.cell
def _(TEXT, nlp, stemmer):
    doc = nlp(TEXT)
    stemmed = [stemmer.stem(token.text) for token in doc if not token.is_punct]
    print(" ".join(stemmed))
    return (doc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · Lemmatization

    Lemmatization reduces words to their **lemma** (dictionary form) using a dictionary
    and POS tags. Unlike stemming, it always produces a real word.
    """)
    return


@app.cell
def _(doc):
    print(f"{'Token':<12} {'POS':<8} {'Lemma'}")
    print("-" * 35)
    for token in doc:
        print(f"{token.text:<12} {token.pos_:<8} {token.lemma_}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Notice how spaCy handles cases that stemming can't:

    - `"'s"` → `"be"` (contraction resolved via POS tag but only for verbs)
    - `"locked"` → `"lock"` (regular past tense)

    spaCy uses POS tags automatically — no manual input needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4 · Stop words

    Stop words are very common words that carry little meaning on their own:
    *the, is, at, which, on, a, an, in, my, he, and, or, but, ...*

    From our sentence, the useful words are things like `"Dre"`, `"locked"`, `"basement"`,
    `"dead"` — not `"and"`, `"in"`, `"my"`, `"he"`.
    """)
    return


@app.cell
def _(nlp):
    print(f"spaCy knows {len(nlp.Defaults.stop_words)} stop words")

    # Show a sample
    print("Examples:", sorted(list(nlp.Defaults.stop_words))[:20])
    return


@app.cell
def _(doc):
    content_tokens = [
        token.text for token in doc if not token.is_stop and not token.is_punct
    ]
    print("Content tokens:", content_tokens)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    That's the core content of the sentence — the words that actually tell you what it's about.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5 · Preprocessing pipeline

    In practice, you chain these steps together. Here's a configurable `preprocess` function
    that runs the full normalization pipeline:
    """)
    return


@app.cell
def _(nlp, stemmer):
    def preprocess(
        text: str,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        remove_punct: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
    ) -> list[str]:
        """
        Configurable text preprocessing pipeline.

        - lowercase: apply case folding
        - remove_stopwords: filter out stop words
        - remove_punct: filter out punctuation tokens
        - lemmatize: reduce tokens to their lemma (spaCy)
        - stem: apply Porter stemming (overrides lemmatize if both are True)
        """
        if lowercase:
            text = text.lower()

        doc = nlp(text)

        tokens = []
        for token in doc:
            if remove_punct and token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue

            if stem:
                tokens.append(stemmer.stem(token.text))
            elif lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return tokens

    return (preprocess,)


@app.cell
def _(TEXT, preprocess):
    print("Default (lowercase + lemma + no stop/punct):")
    print(preprocess(TEXT))
    return


@app.cell
def _(TEXT, preprocess):
    print("With stemming instead of lemmatization:")
    print(preprocess(TEXT, lemmatize=False, stem=True))
    return


@app.cell
def _(TEXT, preprocess):
    print("Minimal (only lowercase):")
    print(
        preprocess(
            TEXT, remove_stopwords=False, remove_punct=False, lemmatize=False
        )
    )
    return


if __name__ == "__main__":
    app.run()
