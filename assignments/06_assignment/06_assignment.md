# Assignment 06
 
Answer these questions in the [06_assignment.py](./assignments/06_assignment/06_assignment.py) marimo notebook or copy and paste it with your team member name or task name.
 
---
 
1. **Semantic song similarity.** Using the `all-MiniLM-L6-v2` embeddings of the song lyrics, build a function `find_similar(song_id: int, n: int = 5) -> DataFrame` that takes the index of a song and returns the `n` most similar songs by cosine similarity, excluding the query song itself. Show columns: title, artist, year, genre, and similarity score. Run it for 3 songs of your choice. Do the results make sense?
2. **Compare to TF-IDF (assignment 04).** Run your embedding-based `find_similar` on the **same 3 songs** you used for the TF-IDF song similarity in assignment 04 (task 2). Put the two top-5 lists side by side and answer the following in a markdown cell:
   - Where do the embedding-based and TF-IDF results agree, and where do they differ?
   - Find at least one song that the embedding search retrieves but TF-IDF does not (or ranks much lower) and vice versa. Look at the lyrics: why might that be? 
   - Which method do you find returns more meaningfully similar songs, and why?
3. **Topics meet metadata.** Using the BERTopic model and topic assignments from the tutorial, connect the discovered topics to the dataset's metadata:
   a) Cross-tabulate **topic vs. genre** (e.g. a `pandas.crosstab` or a heatmap). Do any topics line up clearly with a genre? Are there topics that cut across several genres?
   b) Pick one topic and look at how it is distributed across the years. Briefly comment on whether the topic appears stable over time or concentrated in a particular period.

 