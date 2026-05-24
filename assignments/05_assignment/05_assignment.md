# Assignment 05

Answer these questions in the [05_assignment.py](./assignments/05_assignment/05_assignment.py) marimo notebook or copy and paste it with your team member name or task name.

This assignment covers BERT pre-training and fine-tuning. You will fine-tune two encoder models for genre classification and compare them against each other and against the TF-IDF baseline from assignment 04.

---

1. **Fine-tune ModernBERT.** Fine-tune `answerdotai/ModernBERT-base` on the genre training split using the Hugging Face `Trainer`. Use the same `LyricsDataset` class and training setup as in the session notebook. Report train and test weighted F1 after training.

2. **Fine-tune a second BERT model.** Search [huggingface.co/models](https://huggingface.co/models) for an English encoder model of your choice. Fine-tune it on the same genre split with the same hyperparameters as task 2. Briefly explain why you picked this model.

3. **Comparison.** Plot a grouped bar chart showing F1 per genre for both fine-tuned models. Also show the overall weighted F1 for both models in a second chart. Compare these results to your best weighted F1 from assignment 04. Answer the following questions in a markdown cell:
   - Which genres benefit most from fine-tuned BERT models compared to TF-IDF?
   - Does ModernBERT outperform your chosen model? Where does each model struggle?
   - What could explain any differences you see between the two BERT models?