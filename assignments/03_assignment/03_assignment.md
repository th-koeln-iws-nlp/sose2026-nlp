# Assignment 03

Answer these questions in the [03_assignment.py](./assignments/03_assignment/03_assignment.py) marimo notebook.

Use spaCy's `nlp.pipe` to process the large amount of lyrics. This will take several minutes.

---

1. **Adjective universe.** Extract all adjectives (`token.pos_ == "ADJ"`) from every song, using their lemma form. For each genre, count how often each adjective lemma appears. Then find the **signature adjectives** per genre: adjectives that are disproportionately common in one genre compared to the overall dataset. To do this, for each adjective, compute its share of all adjective uses *within* a genre and its share *across all genres*. The ratio between these two numbers tells you how characteristic an adjective is for that genre. Show the top 5 signature adjectives for at least 3 genres of your choice including their ratio. What do these adjectives reveal about each genre's themes or style?

2. **Geography of lyrics.** Extract all named entities of type `GPE` and `LOC` from every song. Again, each song contributes each unique location only once. Create a table of the 20 most mentioned places with columns: place name, entity type, number of songs mentioning it, and the top 3 genres that mention it. 
   **Bonus:** Plot the top 50 locations on an interactive world map, with marker size proportional to mention count.

3. **Most loved entities.** Use the spaCy `Matcher` to find all spans where a pronoun is followed by a form of "love" and then a named entity (pattern: `PRON` + lemma "love" + token with `ENT_TYPE` set). For each match, extract the entity text and its entity type. Count how often each entity is loved, but each song may only contribute a given entity **once** (deduplicate per song). Show the 15 most loved entities as a bar chart, colored by entity type. What kinds of entities do people sing about loving?


