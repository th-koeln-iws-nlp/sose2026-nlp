import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import re

    import marimo as mo

    # nltk is a classic NLP library with a simple word tokenizer. spaCy is a modern NLP library with a more sophisticated tokenizer.
    import nltk
    import spacy

    nltk.download("punkt_tab", quiet=True)
    # We load the small English model, which includes tokenization rules and exceptions (e.g., known abbreviations like "Dr."). The large model has more features but is slower to load.
    nlp = spacy.load("en_core_web_sm")
    # For model overview see https://spacy.io/models
    return mo, nlp, nltk, re


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tokenization

    Tokenization is the first step in almost every NLP pipeline — splitting raw text
    into meaningful units (tokens). There are many ways to do this, from simple
    string splitting to sophisticated subword algorithms used by modern LLMs.

    We'll compare them all on the same running example:
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
    ## 1 · Naive approaches

    The simplest tokenizers use basic string operations or regex.
    """)
    return


@app.cell
def _(TEXT, re):
    # Approach 1: split on whitespace
    _split_tokens = TEXT.split(" ")
    print("split(' ')     :", _split_tokens)
    print(f"  → {len(_split_tokens)} tokens\n")

    # Approach 2: regex — extract word characters only
    _regex_tokens = re.findall(r"\w+", TEXT)
    print("re.findall(\\w+):", _regex_tokens)
    print(f"  → {len(_regex_tokens)} tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both give the same **count** but very different results:

    - `split(" ")` keeps punctuation attached: `"said…"`, `"nothing,"`, `"idiots!"`
    - `re.findall(r'\w+')` drops all punctuation — cleaner, but we lose information

    Neither handles abbreviations (`Dr.`) or contractions (`Dre's`, `he's`) well.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · NLTK tokenization
    """)
    return


@app.cell
def _(TEXT, nltk):
    _nltk_tokens = nltk.word_tokenize(TEXT)
    print("NLTK:", _nltk_tokens)
    print(f"  → {len(_nltk_tokens)} tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    NLTK's `word_tokenize` uses the Penn Treebank tokenization rules.
    It splits punctuation and contractions but keeps `Dr.` together as a known abbreviation.
    However, it doesn't split `said…` — the ellipsis stays attached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · spaCy tokenization

    This is the first time we process text with spaCy. We just call the nlp pipeline on the text and iterate over the resulting tokens. spaCy applies a combination of rules and exceptions to split the text into tokens.
    """)
    return


@app.cell
def _(TEXT, nlp):
    doc = nlp(TEXT)

    for _token in doc:
        print(_token.text)
    return (doc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    What happened:

    - `"said…"` → `"said"` + `"…"` (suffix rule split the ellipsis)
    - `"nothing,"` → `"nothing"` + `","` (suffix rule split the comma)
    - `"idiots!"` → `"idiots"` + `"!"` (suffix rule split the exclamation)
    - `"Dr."` stays as one token (exception rule: known abbreviation)
    - "Dre's" → "Dre" + "'s" (possessive split off)
    - "he's" → "he" + "'s" (contraction split off)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 · Useful spaCy token attributes

    Once you have tokens, each one carries information:
    """)
    return


@app.cell
def _(doc):
    print(
        f"{'Text':<10} {'is_alpha':<10} {'is_punct':<10} {'like_num':<10} {'is_stop':<10} {'lemma_'}"
    )
    print("-" * 60)
    for _token in doc:
        print(
            f"{_token.text:<10} {str(_token.is_alpha):<10} {str(_token.is_punct):<10} {str(_token.like_num):<10} {str(_token.is_stop):<10} {_token.lemma_}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4 · The comparison

    Same sentence, four tokenizers:
    """)
    return


@app.cell
def _(TEXT, mo, nlp, nltk, re):
    _methods = {
        'split(" ")': TEXT.split(" "),
        r"re.findall(r'\w+')": re.findall(r"\w+", TEXT),
        "NLTK word_tokenize": nltk.word_tokenize(TEXT),
        "spaCy": [t.text for t in nlp(TEXT)],
    }

    _rows = [
        {"Method": name, "Tokens": " · ".join(tokens), "Count": len(tokens)}
        for name, tokens in _methods.items()
    ]

    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `split` and `\w+` give the same count but different results — one keeps punctuation
    attached, the other drops it. NLTK and spaCy give similar results but differ on
    edge cases (e.g., handling of `…`, URLs, or `U.K.`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.1 · BERT tokenizer (WordPiece)
    """)
    return


@app.cell
def _(TEXT):
    from transformers import BertTokenizer

    _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    _tokens = _tokenizer.tokenize(TEXT)
    print("BERT tokens:", _tokens)
    print(f"  → {len(_tokens)} tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - The `##` prefix means: *"this is a continuation, not a word start"*
    - Common words like `"and"`, `"nothing"` stay whole
    - Rare or compound words get split
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.2 · GPT tokenizer (BPE via tiktoken)

    GPT models use **Byte Pair Encoding (BPE)**. The `tiktoken` library
    is OpenAI's fast BPE tokenizer.
    """)
    return


@app.cell
def _(TEXT):
    import tiktoken

    _enc = tiktoken.encoding_for_model("gpt-4")
    _token_ids = _enc.encode(TEXT)
    _tokens = [_enc.decode([t]) for t in _token_ids]

    print("GPT-4 tokens:", _tokens)
    print(f"  → {len(_tokens)} tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Notice:
    - Tokens often start with a space (` Dr`, ` Dre`) — this encodes word boundaries
    - Contractions are split: `Dre's` → `Dre` + `'s`
    - Common words stay whole, rare ones may be split into subwords
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6 · Regex for pre-tokenization

    Before running BPE, text is first **pre-tokenized** — roughly split by whitespace
    and punctuation using regex. Here's a simplified version of the GPT-2 pre-tokenizer:
    """)
    return


@app.cell
def _(TEXT, re):
    # Simplified GPT-2 pre-tokenizer using the built-in `re` module
    # The real GPT-2 uses the `regex` library for Unicode properties like \p{L}
    _gpt2_pattern = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d"  # contractions
        r"| ?[a-zA-Z]+"  # words (with optional leading space)
        r"| ?[0-9]+"  # numbers
        r"| ?[^\s\w]+"  # punctuation
        r"|\s+",  # whitespace
    )

    _tokens = _gpt2_pattern.findall(TEXT)

    print(f"Pre-tokenized: {_tokens}")
    print(f"  → {len(_tokens)} tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how:
    - `Dre's` → `Dre` + `'s` (contraction split)
    - Punctuation becomes its own token: `!`, `,`
    - Some tokens start with a space (` Dr`, ` Dre`) — this encodes word boundaries

    The real GPT-2 pre-tokenizer uses the **`regex`** library (not the built-in `re`)
    with Unicode properties like `\p{L}` (any letter) and `\p{N}` (any number).
    Above is a simplified version using just `re`.
    """)
    return


if __name__ == "__main__":
    app.run()
