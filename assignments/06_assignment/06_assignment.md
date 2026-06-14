# Assignment 06

Answer these questions in the [06_assignment.py](./assignments/06_assignment/06_assignment.py) marimo notebook or copy and paste it with your team member name or task name.

Keep only **genres with at least 200 songs**. From those, draw a **stratified random test sample of about 500 songs** (set a random seed so it is reproducible), and keep the remaining labeled songs as a pool to draw few-shot examples from. Those genres are your **label set**. Classify **one song per request** (we cover batching in the next session). Pick **one** model and use it throughout:

- **Gemini 3 Flash Preview** (API), or
- a self-hosted model from Open WebUI: **Mistral 3.2**, **Qwen 3.5**, or **Qwen 3.6**

Set **temperature 0** everywhere (reproducible and comparable), and truncate each song's lyrics to the **first 1,500 characters** before sending. Reuse the model client setup from the session tutorial notebook.

---

1. **Zero-shot genre classification.** Write a **system prompt** that turns the model into a genre classifier: give it the role, the **exact allowed genre labels** (your label set), and the instruction to answer with **one label only**. Send the truncated lyrics as the user prompt. Parse the response to a single label and **validate that it is in the allowed set** (map anything else to `unknown`). Run on the test sample and report **weighted F1**.

2. **Few-shot genre classification.** Add a small number of labeled examples (for example 1 per genre) drawn from the **example pool** to your prompt, keeping everything else identical. Re-run on the same test sample and report **weighted F1**. In a markdown cell, state whether few-shot helped and by how much.

3. **Token and credit usage.** Report the total tokens and the cost in credits / euros for both the zero-shot and few-shot runs, the same way as in the session tutorial.

4. **Comparison.** Plot a grouped bar chart of **F1 per genre** for your zero-shot run next to your few-shot run, and report the **overall weighted F1** for both. Also compare against your **best classifiers from the previous assignments** (TF-IDF from Assignment 04 and your best fine-tuned BERT from Assignment 05): add their overall weighted F1 to the comparison. In a markdown cell, briefly answer: which genres are hardest for the model, did few-shot improve the same genres or different ones, and how does the LLM stack up against your trained classifiers?

*Remark: the set sizes differ (this assignment uses a ~500-song sample with genres of at least 200 songs, the earlier assignments used their own splits), so this is a rough proxy, not a like-for-like comparison. It still tells you roughly where a prompted LLM lands relative to a model you trained yourself.*