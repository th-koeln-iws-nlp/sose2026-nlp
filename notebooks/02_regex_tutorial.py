import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import re

    import marimo as mo
    import pandas as pd

    return mo, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Regular Expressions for NLP

    Regular expressions (regex) are one of the most useful tools for text processing.
    They let us **search**, **extract**, and **transform** text using patterns — essential
    skills for working with messy real-world data like song lyrics.

    > *"Formally, a regular expression is an algebraic notation for characterizing a set of strings."*
    > — Jurafsky & Martin, *Speech and Language Processing*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Basic Building Blocks

    | Pattern | Matches | Example |
    |---------|---------|---------|
    | `.` | Any single character | `c.t` → "cat", "cot", "cut" |
    | `\d` | Any digit (0–9) | `\d\d` → "42" |
    | `\w` | Letter, digit, or underscore | `\w+` → "hello_42" |
    | `\s` | Whitespace (space, tab, newline) | `\s+` → `"   "` |
    | `\b` | Word boundary | `\bcat\b` → "cat" but not "caterpillar" |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Quantifiers

    | Pattern | Meaning | Example |
    |---------|---------|---------|
    | `*` | Zero or more | `ha*` → "h", "ha", "haa", "haaa" |
    | `+` | One or more | `ha+` → "ha", "haa", "haaa" (not "h") |
    | `?` | Zero or one (optional) | `colou?r` → "color", "colour" |
    | `{n}` | Exactly n times | `\d{4}` → "2024" |
    | `{n,m}` | Between n and m | `\d{1,3}` → "1", "42", "999" |

    By default, quantifiers are **greedy** (match as much as possible).
    Add `?` for **non-greedy**: `.*?` matches as little as possible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Character Classes and Groups

    | Pattern | Meaning |
    |---------|---------|
    | `[aeiou]` | Any vowel |
    | `[A-Z]` | Any uppercase letter |
    | `[0-9]` | Any digit (same as `\d`) |
    | `[^aeiou]` | Any character that is NOT a vowel |
    | `(cat\|dog)` | "cat" or "dog" |
    | `(?:...)` | Non-capturing group (grouping without storing) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Python `re` Module — The Essentials
    """)
    return


@app.cell
def _(re):
    _text = "And Dr. Dre said… nothing, you idiots! Dr. Dre's dead, he's locked in my basement (ha ha!)"

    # FIND all matches
    _words = re.findall(r"\w+", _text)
    print("findall(r'\\w+'):", _words)

    # REPLACE
    _normalized = re.sub(r"\u2026", "...", _text)  # replace … with ...
    print("sub(…→...)     :", _normalized)

    # SPLIT
    _tokens = re.split(r"\s+", _text)
    print("split(r'\\s+')  :", _tokens)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the difference between `findall(r'\w+')` and `split(r'\s+')`:

    - `findall` extracts only word characters → punctuation is **lost**
    - `split` keeps punctuation attached to words → "said…" stays as one token

    Trade-offs everywhere! This is exactly why tokenization is a hard problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5 · Regex on our Lyrics Dataset

    Let's load the Billboard Top 100 lyrics and explore them with regex.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df = df.rename(
        columns={
            "Lyrics": "lyrics_raw",
            "Artist": "artist",
            "Song Title": "song_title",
            "Year": "year",
        }
    )
    print(f"Loaded {len(df):,} songs")
    df[["artist", "song_title", "year", "lyrics_raw"]].head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.1 · Section markers in raw lyrics

    Raw lyrics from Genius contain section markers like `[Verse 1]`, `[Chorus]`, `[Bridge]`, etc.
    Let's find them.
    """)
    return


@app.cell
def _(df, re):
    # Pick one song to demonstrate
    _example = df.loc[0, "lyrics_raw"]
    _song = f"{df.loc[0, 'artist']} – {df.loc[0, 'song_title']}"

    print(f"🎵 {_song}\n")
    print("Raw lyrics (first 500 chars):")
    print(_example[:500])
    print("\n--- Section markers found ---")
    _markers = re.findall(r"\[.*?\]", _example)
    print(_markers)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pattern `\[.*?\]` means:
    - `\[` — literal opening bracket (escaped because `[` is special in regex)
    - `.*?` — any characters, **non-greedy** (stop at the first `]`)
    - `\]` — literal closing bracket

    Without the `?` (i.e., `\[.*\]`), the greedy `.*` would match everything from the
    **first** `[` to the **last** `]` in the text — not what we want!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.2 · What section types exist in our dataset?

    Let's find **all unique section markers** across the entire dataset.
    """)
    return


@app.cell
def _(df, pd, re):
    _all_markers = (
        df["lyrics_raw"].dropna().apply(lambda t: re.findall(r"\[.*?\]", t))
    )
    _flat = [m for _markers in _all_markers for m in _markers]

    # Strip brackets, remove non-alphanumeric (keep spaces), normalize whitespace, title-case
    _normalized = [
        re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", m.strip("[]"))).strip().title()
        for m in _flat
    ]
    # Group by base type (remove numbers)
    _base = [re.sub(r"\s*\d+\s*", "", m).strip() for m in _normalized]

    marker_counts = pd.Series(_base).value_counts().head(20)
    print(f"Found {len(_flat):,} section markers total\n")
    marker_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.3 · Cleaning lyrics with regex

    This allows us to clean the raw lyrics like this:
    """)
    return


@app.cell
def _(df, re):
    _raw = df.loc[0, "lyrics_raw"]
    _song = f"{df.loc[0, 'artist']} – {df.loc[0, 'song_title']}"

    # Step 1: Remove section markers like [Verse 1], [Chorus]
    _step1 = re.sub(r"\[.*?\]", "", _raw, flags=re.DOTALL)

    # Step 2: Collapse multiple newlines into two
    _step2 = re.sub(r"\n{3,}", "\n\n", _step1)

    # Step 3: Collapse multiple spaces into one
    _step3 = re.sub(r" {2,}", " ", _step2).strip()

    print(f"🎵 {_song}\n")
    print("=== BEFORE (first 400 chars) ===")
    print(_raw[:400])
    print("\n=== AFTER (first 400 chars) ===")
    print(_step3[:400])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What each regex does:**

    | Step | Pattern | Meaning |
    |------|---------|---------|
    | 1 | `\[.*?\]` | Match `[anything]` — removes section markers |
    | 2 | `\n{3,}` | Three or more newlines → collapse to two |
    | 3 | ` {2,}` | Two or more spaces → collapse to one |

    The `re.DOTALL` flag makes `.` also match newlines, so markers that
    span multiple lines (rare but possible) are also caught.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6 · Pattern Extraction

    Regex isn't just for cleaning. It is also great for **finding patterns** in text.
    """)
    return


@app.cell
def _(df, re):
    # Work with one lyric
    _lyrics = df.loc[0, "lyrics_raw"]
    _song = f"{df.loc[0, 'artist']} – {df.loc[0, 'song_title']}"
    print(f"🎵 {_song}\n")

    # Find all contractions (words with apostrophe)
    _contractions = re.findall(r"\b\w+'\w+\b", _lyrics)
    print("Contractions found:", sorted(set(_contractions)))

    # Find all words in ALL CAPS (2+ letters)
    _caps = re.findall(r"\b[A-Z]{2,}\b", _lyrics)
    print("ALL CAPS words:   ", sorted(set(_caps)) if _caps else "(none)")

    # Find parenthetical content
    _parens = re.findall(r"\(([^)]+)\)", _lyrics)
    print("Parentheticals:   ", sorted(set(_parens)) if _parens else "(none)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.1 · Finding repeated words across the dataset

    Repeated words like "yeah yeah yeah" or "na na na" are common in lyrics.
    Let's use a **capture group with a backreference** to find them.
    """)
    return


@app.cell
def _(df, pd, re):
    def _find_repetitions(text):
        if not isinstance(text, str):
            return []
        # finditer instead of findall to match the entire string
        return [
            m.group()
            for m in re.finditer(r"\b(\w+)(?:\s+\1){2,}\b", text, re.IGNORECASE)
        ]


    _sample = df.head(200).copy()
    _sample["repetitions"] = _sample["lyrics"].apply(_find_repetitions)
    _with_reps = _sample[_sample["repetitions"].str.len() > 0][
        ["artist", "song_title", "repetitions"]
    ]

    print(f"Songs with repeated words (first 200 songs): {len(_with_reps)}\n")
    pd.set_option("display.max_colwidth", 80)
    _with_reps.head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How the repetition pattern works:**

    `\b(\w+)(?:\s+\1){2,}\b`

    - `\b` — word boundary
    - `(\w+)` — capture group: match and **store** a word
    - `(?:\s+\1){2,}` — non-capturing group repeated 2+ times:
      - `\s+` — whitespace
      - `\1` — **backreference** to the captured word (must be the same word)
    - `\b` — word boundary

    So this matches a word followed by **at least 2 more** copies of itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.2 · The same pattern with `pregex`

    The `pregex` package lets you build regex patterns programmatically —
    no more deciphering cryptic pattern strings.
    """)
    return


@app.cell
def _():
    from pregex.core.classes import AnyWordChar
    from pregex.core.quantifiers import OneOrMore, AtLeast
    from pregex.core.groups import Backreference, Capture
    from pregex.core.assertions import WordBoundary

    # Build the same \b(\w+)(?:\s+\1){2,}\b pattern with pregex
    _word = Capture(OneOrMore(AnyWordChar()))
    _repetition = AtLeast(OneOrMore(" ") + Backreference(1), 2)
    _pattern = WordBoundary() + _word + _repetition + WordBoundary()

    print("Generated regex:", _pattern.get_pattern())
    print("Matches:", _pattern.get_matches("yeah yeah yeah test na na na na"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.3 · Capture groups for data extraction

    Capture groups store matched text so we can reuse it.
    Useful for extracting structured information from unstructured text.
    """)
    return


@app.cell
def _(re):
    # Example: swap US date format (mm/dd/yyyy) to EU format (dd-mm-yyyy)
    _us_dates = "Released on 03/15/2019 and re-released 11/02/2020"

    _eu_dates = re.sub(r"(\d{2})/(\d{2})/(\d{4})", r"\2-\1-\3", _us_dates)

    print("US format:", _us_dates)
    print("EU format:", _eu_dates)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `(\d{2})/(\d{2})/(\d{4})` creates three capture groups:
    - `\1` = month, `\2` = day, `\3` = year

    In the replacement `\2-\1-\3`, we rearrange them to day-month-year.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7 · Python Packages for Pattern Matching

    Beyond the built-in `re` module, there are several useful packages:

    | Package | What it does |
    |---------|-------------|
    | **`re`** (built-in) | Standard regex — sufficient for most tasks |
    | **[`regex`](https://github.com/mrabarnett/mrab-regex)** | Drop-in replacement for `re` with Unicode properties (`\p{L}`), fuzzy matching, and more |
    | **[`pregex`](https://github.com/manoss96/pregex)** | Build regex patterns programmatically — no more unreadable pattern strings |
    | **[`pyparsing`](https://github.com/pyparsing/pyparsing)** | Full parser library for complex grammars — when regex isn't enough |

    For a practical comparison, see the
    [Text Pattern Matching Guide](https://codecut.ai/regex-pregex-pyparsing-text-pattern-matching-guide/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 8 · Useful Resources

    - **[regex101.com](https://regex101.com/)** — Interactive regex tester with real-time
      explanations. Select "Python" flavor. Paste your pattern and text to debug.
    - **[Jurafsky & Martin, Ch. 2.6](https://web.stanford.edu/~jurafsky/slp3/2.pdf)** — Comprehensive
      regex reference in an NLP context
    - **[Text Pattern Matching Guide](https://codecut.ai/regex-pregex-pyparsing-text-pattern-matching-guide/)**
      — Comparing `regex`, `pregex`, and `pyparsing`
    """)
    return


if __name__ == "__main__":
    app.run()
