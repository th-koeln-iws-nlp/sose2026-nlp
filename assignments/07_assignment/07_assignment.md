# Assignment 07

Answer these questions in the [07_assignment.py](./assignments/07_assignment/07_assignment.py) marimo notebook or copy and paste it with your team member name or task name.

Keep only songs with a non-empty `niche_genres` list. Draw a **random test sample of about 500 songs** (set a seed so it is reproducible), and keep the remaining labeled songs as a pool for examples and tag statistics. Pick **one** model and use it throughout:

- **Gemini 3 Flash Preview** (API), or
- a self-hosted model from Open WebUI: **Mistral 3.2**, **Qwen 3.5**, or **Qwen 3.6**

Set **temperature 0**, truncate each song's lyrics to the **first 1,500 characters**, and have the model return its answer as a **JSON array of tags**, validated as in the session tutorial. One request per song.

---

1. **Explore the label space.** Using the example pool, report the **number of distinct niche tags** and list the **top 50 most frequent** ones. This top 50 set is your **allowed tag set** for scoring. Report **coverage**: the share of the test sample's true tags that fall inside it. 

2. **Zero-shot prediction.** Write a **system prompt** that asks for a song's niche subgenres as a **JSON array** of up to 3 tags, from the lyrics, with no examples. Score against the true `niche_genres` with  **mean Jaccard** (set overlap per song, averaged). Normalize tags to lowercase before matching. 

3. **Few-shot prediction.** Add a handful of worked examples (lyrics with their correct tags) drawn from the **example pool**, keeping everything else identical. Re-run on the same test sample and report the same metric. In a markdown cell, state whether few-shot helped and by how much.

4. **Token and credit usage.** Report the total tokens and the cost in credits / euros for both runs, the same way as in the session tutorial.

*Remark: Spotify's niche tags are noisy and assigned at the artist level, not the song level, so a "wrong" prediction is sometimes the model being more right than the label. Treat these scores as a rough signal, not ground truth.*

**Optional.** Send the songs in **batches** instead of one per request, returning a JSON array keyed by song id (as in the lecture). Re-request any ids the model drops. 