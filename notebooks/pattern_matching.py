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
    from spacy.matcher import Matcher, PhraseMatcher

    nlp = spacy.load("en_core_web_sm")
    df = (
        pd.read_csv(Path("data/billboard_top100/billboard_top_100.csv"))
        .dropna(subset=["lyrics"])
        .reset_index(drop=True)
    )
    return Counter, Matcher, PhraseMatcher, df, mo, nlp, pd, px, spacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rule-Based Pattern Matching in spaCy

    spaCy's statistical NER model generalizes well, but it misses domain-specific
    entities and makes mistakes in unusual registers like song lyrics.

    **Rule-based approaches** complement statistical models: if you *know* what you're
    looking for, you can match it exactly.

    spaCy gives you three tools for rule-based matching:

    | Tool | Best for |
    |------|----------|
    | **Matcher** | Complex patterns using linguistic features (POS, dep, shape, ...) |
    | **PhraseMatcher** | Fast lookup of large term lists |
    | **EntityRuler** | Adding rule-based entities directly into the pipeline |

    ---

    ## Part 1: The Matcher

    The `Matcher` finds **token sequences** based on rules you define.
    Think of it as **regex, but for linguistic annotations** — you can match on POS tags,
    dependency labels, word shape, lemmas, not just characters.

    A pattern is a **list of dictionaries**. Each dictionary describes **one token.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Token attributes you can match on

    | Attribute | Matches on | Example |
    |-----------|-----------|---------|
    | `TEXT` | Exact text | `{"TEXT": "Apple"}` |
    | `LOWER` | Lowercase form | `{"LOWER": "apple"}` — matches "Apple", "APPLE", "apple" |
    | `LEMMA` | Base form | `{"LEMMA": "be"}` — matches "is", "was", "are", "been" |
    | `POS` | UPOS tag | `{"POS": "NOUN"}` |
    | `TAG` | Penn Treebank tag | `{"TAG": "VBG"}` — gerund/present participle |
    | `DEP` | Dependency label | `{"DEP": "nsubj"}` |
    | `SHAPE` | Word shape | `{"SHAPE": "Xxxxx"}` — capitalized 5-letter word |
    | `IS_ALPHA` | Alphabetic? | `{"IS_ALPHA": True}` |
    | `IS_DIGIT` | All digits? | `{"IS_DIGIT": True}` |
    | `IS_PUNCT` | Punctuation? | `{"IS_PUNCT": True}` |
    | `IS_STOP` | Stop word? | `{"IS_STOP": False}` |
    | `LIKE_NUM` | Number-like? | `{"LIKE_NUM": True}` — matches "10", "ten", "3.5" |
    | `LENGTH` | Token length | `{"LENGTH": {">=": 10}}` |
    | `ENT_TYPE` | Entity type | `{"ENT_TYPE": "PERSON"}` |

    The **most common mistake:** putting multi-word text in a single dict.

    ```python
    # WRONG — tries to match a single token with text "New York"
    pattern = [{"TEXT": "New York"}]

    # CORRECT — two tokens, one dict each
    pattern = [{"TEXT": "New"}, {"TEXT": "York"}]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Operators: controlling repetition

    Just like regex quantifiers, Matcher patterns support **operators:**

    | OP | Meaning | Regex equivalent |
    |----|---------|-----------------|
    | `"!"` | Match 0 times (negation) | — |
    | `"?"` | Match 0 or 1 times | `?` |
    | `"+"` | Match 1 or more times | `+` |
    | `"*"` | Match 0 or more times | `*` |

    ```python
    # "iPhone" optionally followed by a number
    pattern = [
        {"LOWER": "iphone"},
        {"IS_DIGIT": True, "OP": "?"}
    ]
    # Matches: "iPhone", "iPhone 15", "IPHONE 7"
    ```

    ### Set membership with IN / NOT_IN

    ```python
    # "good", "great", or "excellent" followed by a noun
    pattern = [
        {"LOWER": {"IN": ["good", "great", "excellent"]}},
        {"POS": "NOUN"}
    ]

    # Any verb that is NOT "be" or "have"
    pattern = [
        {"POS": "VERB", "LEMMA": {"NOT_IN": ["be", "have"]}}
    ]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Part 2: Matching "I [VERB] [NOUN]" in lyrics

    Pick a song and run the `Matcher` on its lyrics to find **"I + VERB + (DET) + NOUN"** patterns —
    a classic structure in pop music that reveals what the singer claims to do.
    The determiner is optional (`OP: "?"`) so the pattern matches both "I love music" and "I kissed a girl".
    """)
    return


@app.cell
def _(df, mo):
    song_options = {
        f"{row['artist']} - {row['song_title']}": i for i, row in df.iterrows()
    }
    song_picker = mo.ui.dropdown(
        options=song_options,
        value=list(song_options.keys())[0],
        label="Pick a song",
        full_width=True,
    )
    song_picker
    return (song_picker,)


@app.cell
def _(Matcher, df, nlp, song_picker):
    lyrics = df.loc[song_picker.value, "lyrics"]
    doc = nlp(lyrics)

    i_verb_noun_matcher = Matcher(nlp.vocab)
    i_verb_noun_matcher.add(
        "I_VERB_NOUN",
        [
            [
                {"LOWER": "i"},
                {"POS": "VERB"},
                {"POS": "DET", "OP": "?"},
                {"POS": "NOUN"},
            ]
        ],
    )

    i_verb_noun_matches = [
        doc[_s:_e].text for _, _s, _e in i_verb_noun_matcher(doc)
    ]
    return doc, i_verb_noun_matches


@app.cell
def _(i_verb_noun_matches, mo):
    mo.md(
        '**"I VERB NOUN" matches:**\n\n'
        + (
            "\n".join(f"- `{match}`" for match in i_verb_noun_matches)
            if i_verb_noun_matches
            else "_No matches found in this song._"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Part 3: Adjective chains before a noun

    Lyrics often stack adjectives before a noun for emotional emphasis.
    Let's match **one or more adjectives followed by a noun** using the `+` operator.
    """)
    return


@app.cell
def _(Matcher, doc, mo, nlp):
    adj_noun_matcher = Matcher(nlp.vocab)
    adj_noun_matcher.add(
        "ADJ_NOUN",
        [
            [
                {"POS": "ADJ", "OP": "+"},
                {"POS": "NOUN"},
            ]
        ],
    )
    adj_noun_phrases = list(
        dict.fromkeys(doc[_s:_e].text for _, _s, _e in adj_noun_matcher(doc))
    )
    mo.md(
        "**ADJ+ NOUN matches (unique):**\n\n"
        + (
            "\n".join(f"- `{phrase}`" for phrase in adj_noun_phrases[:20])
            if adj_noun_phrases
            else "_No adjective–noun phrases found._"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 4: Corpus-wide pattern frequency

    Let's count **"I [VERB] (DET) [NOUN]"** matches across **all** songs.
    We use `nlp.pipe` to process lyrics efficiently in batches.
    """)
    return


@app.cell
def _(Counter, Matcher, df, nlp, pd):
    corpus_matcher = Matcher(nlp.vocab)
    corpus_matcher.add(
        "I_VERB_NOUN",
        [
            [
                {"LOWER": "i"},
                {"POS": "VERB"},
                {"POS": "DET", "OP": "?"},
                {"POS": "NOUN"},
            ]
        ],
    )

    phrase_counts = Counter()
    for song_doc in nlp.pipe(df["lyrics"].dropna().tolist(), batch_size=128):
        for _, _s, _e in corpus_matcher(song_doc):
            phrase_counts[song_doc[_s:_e].text.lower()] += 1

    corpus_match_df = pd.DataFrame(
        [
            {"phrase": phrase, "count": count}
            for phrase, count in phrase_counts.most_common(20)
        ]
    )
    return (corpus_match_df,)


@app.cell
def _(corpus_match_df, px):
    px.bar(
        corpus_match_df,
        x="phrase",
        y="count",
        text="count",
        labels={"phrase": "", "count": "occurrences"},
        title='Top 20 "I VERB (DET) NOUN" phrases across all Billboard songs',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 5: The PhraseMatcher

    The `Matcher` is powerful but slow for large vocabularies — you'd need to write each
    token pattern by hand.

    The **PhraseMatcher** is optimized for matching **long lists of exact phrases:**

    ```python
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab)
    cities = ["New York", "Los Angeles", "Cologne", "Berlin"]
    patterns = [nlp.make_doc(city) for city in cities]
    matcher.add("CITIES", patterns)
    ```

    `nlp.make_doc()` creates a `Doc` *without* running the full pipeline — fast.

    **When to use which tool:**

    | Scenario | Use |
    |----------|-----|
    | 10 patterns with linguistic features | **Matcher** |
    | 500 artist names from a spreadsheet | **PhraseMatcher** |
    | "adjective + noun" abstract pattern | **Matcher** |
    | City / drug / genre names from a list | **PhraseMatcher** |

    Rule of thumb: if you have a **list**, use PhraseMatcher.
    If you have a **pattern**, use Matcher.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### PhraseMatcher: find artist name mentions across lyrics

    Many songs reference other artists. Let's build a PhraseMatcher from the artist
    names in the dataset and see who gets name-dropped most often.

    We restrict to **multi-word artist names** (e.g. "Katy Perry", "The Beatles") —
    single-token names like "Time", "Who", or "People" are too ambiguous and would
    produce false matches on common words. Multi-word spans are exactly where
    PhraseMatcher shines over a plain string search.
    """)
    return


@app.cell
def _(Counter, PhraseMatcher, df, nlp, pd):
    # Only keep multi-word artist names to avoid false matches on common single words
    # like "Time", "Who", "People"
    artist_names = [
        name for name in df["artist"].dropna().unique() if len(name.split()) > 1
    ]
    artist_phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    artist_phrase_matcher.add(
        "ARTIST", [nlp.make_doc(name) for name in artist_names]
    )

    mention_counts = Counter()
    for song_lyrics, song_artist in zip(
        nlp.pipe(df["lyrics"].dropna().tolist(), batch_size=128),
        df["artist"].dropna().tolist(),
    ):
        for _, _s, _e in artist_phrase_matcher(song_lyrics):
            mentioned_name = song_lyrics[_s:_e].text
            if mentioned_name.lower() != song_artist.lower():
                mention_counts[mentioned_name] += 1

    artist_mention_df = pd.DataFrame(
        [
            {"artist_mentioned": name, "count": count}
            for name, count in mention_counts.most_common(15)
        ]
    )
    return (artist_mention_df,)


@app.cell
def _(artist_mention_df, px):
    px.bar(
        artist_mention_df,
        x="artist_mentioned",
        y="count",
        text="count",
        labels={"artist_mentioned": "", "count": "mentions in other songs"},
        title="Top 15 artist name-drops across Billboard lyrics (PhraseMatcher)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 6: The EntityRuler

    The `Matcher` and `PhraseMatcher` find spans but **don't add them to `doc.ents`**.

    The **EntityRuler** is a pipeline component that matches patterns AND writes them
    as named entities,accessible via `doc.ents` just like statistical NER results.

    ```python
    ruler = nlp.add_pipe("entity_ruler")

    patterns = [
        {"label": "ORG",    "pattern": "TH Köln"},
        {"label": "GENRE",  "pattern": "Hip-Hop"},
        {"label": "PERSON", "pattern": [{"LOWER": "ines"}, {"LOWER": "montani"}]},
    ]
    ruler.add_patterns(patterns)
    ```

    The EntityRuler accepts **phrase patterns** (strings) and **token patterns**
    (list of dicts), so the same syntax as the Matcher.

    ### Pipeline order matters

    | Position | Behaviour |
    |----------|-----------|
    | `before="ner"` | Rules run first; statistical NER won't overwrite them → use for **known entities** |
    | after NER (default) | Statistical NER runs first; rules fill **gaps** → use to extend coverage |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### EntityRuler: tag music genres in lyrics

    Let's add a custom `GENRE` entity type. spaCy's default model has no concept of
    music genres. We place the EntityRuler **after** the statistical NER so it fills gaps.
    """)
    return


@app.cell
def _(doc, mo, spacy):
    genre_nlp = spacy.load("en_core_web_sm")
    genre_ruler = genre_nlp.add_pipe("entity_ruler")
    genre_ruler.add_patterns(
        [
            {"label": "GENRE", "pattern": name}
            for name in [
                "Hip-Hop",
                "Hip Hop",
                "R&B",
                "R & B",
                "Rock",
                "Pop",
                "Jazz",
                "Blues",
                "Soul",
                "Funk",
                "Reggae",
                "Country",
                "EDM",
                "Trap",
                "Gospel",
                "Disco",
                "Punk",
                "Metal",
                "Indie",
                "Folk",
            ]
        ]
    )

    tagged_doc = genre_nlp(doc.text)
    found_entities = [(ent.text, ent.label_) for ent in tagged_doc.ents]

    mo.md(
        "**Entities found in selected song:**\n\n"
        + (
            "\n".join(
                f"- `{text}` → **{label}**" for text, label in found_entities
            )
            if found_entities
            else "_No entities found._"
        )
    )
    return


if __name__ == "__main__":
    app.run()
