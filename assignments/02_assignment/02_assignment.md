# Assignment 02

Answer these questions in the [02_assignment.py](./assignments/02_assignment/02_assignment.py) marimo notebook.

Use the `preprocess()` function from the [normalization notebook](./notebooks/normalization.py) to tokenize, clean, and lemmatize lyrics.

---

1. **Top 20 words.** Apply the preprocessing pipeline to all lyrics in the dataset. Create two bar charts side by side: the 20 most frequent words **with** stop words and the 20 most frequent words **without** stop words. What changes? Briefly comment on what you observe.

2. **Who talks about love?** Calculate for each song the percentage of tokens that are the lemma "love". Which genre has the highest average love-percentage per song? Visualize the result as a bar chart across genres.

3. **Vocabulary richness.** The [Pudding](https://pudding.cool/projects/vocabulary/index.html) visualized the vocabulary size of popular rappers. Let's do something similar: for each artist with at least 3 songs in the dataset, calculate the **number of unique word types** and the **type-token ratio** (unique types / total tokens) from their preprocessed lyrics. Show the top 10 artists with the largest vocabulary and the top 10 artists with the highest type-token ratio. Are they the same artists? Why might these two rankings differ?
