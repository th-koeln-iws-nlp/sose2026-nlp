import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from torch.utils.data import Dataset

    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Dataset,
        LogisticRegression,
        TfidfVectorizer,
        Trainer,
        TrainingArguments,
        classification_report,
        f1_score,
        go,
        mo,
        np,
        os,
        pd,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decade Classification: ModernBERT vs TF-IDF + Logistic Regression

    Can a model predict which decade a Billboard Top 100 song is from using only its lyrics?
    We compare two approaches on the same 80/20 train/test split:

    | Approach | Features |
    |---|---|
    | TF-IDF + Logistic Regression | Sparse bag-of-words (10k terms) |
    | ModernBERT-base (fine-tuned) | Contextual embeddings |

    Evaluation metric: **weighted F1** across 6 decade classes (1960s - 2010s).
    """)
    return


@app.cell
def _(pd, train_test_split):
    df_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df = df_raw[(df_raw["year"] >= 1960) & (df_raw["year"] <= 2019)].copy()
    df["decade"] = df["year"].apply(lambda y: f"{(y // 10) * 10}s")
    df = df.dropna(subset=["lyrics"]).reset_index(drop=True)

    DECADES = sorted(df["decade"].unique())
    label2id = {d: i for i, d in enumerate(DECADES)}
    id2label = {i: d for d, i in label2id.items()}
    df["label"] = df["decade"].map(label2id)

    all_texts = df["lyrics"].tolist()
    all_labels = df["label"].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(
        f"{len(df)} songs | {len(DECADES)} decades | "
        f"Train: {len(train_texts)} | Test: {len(test_texts)}"
    )
    print("Classes:", DECADES)
    return (
        df,
        id2label,
        label2id,
        test_labels,
        test_texts,
        train_labels,
        train_texts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Device

    Transformers are excellent go be run on a GPU. We can do this by selecting the device `cuda`. If you have a MacBook Pro with a Silicon Valley chip you can also select `mps`.
    """)
    return


@app.cell
def _(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_label = "MPS (Apple Silicon Chip)"
    else:
        device = torch.device("cpu")
        device_label = "CPU"
    device_info = mo.callout(mo.md(f"Device: **{device_label}**"), kind="info")
    device_info
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ModernBERT Fine-tuning

    [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) is a 2024 encoder-only
    transformer trained on 2 trillion tokens with a context window of 8192.
    We add a classification head and fine-tune all weights with the Hugging Face `Trainer`.
    """)
    return


@app.cell
def _(mo):
    mode_radio = mo.ui.radio(
        options=["Train from scratch", "Continue training"],
        value="Train from scratch",
        label="Mode",
    )
    lr_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2, label="Learning rate (×1e-5)"
    )
    batch_slider = mo.ui.slider(
        start=16, stop=64, step=16, value=32, label="Batch size"
    )
    epoch_slider = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epochs")
    max_len_slider = mo.ui.slider(
        start=256, stop=1024, step=256, value=512, label="Max token length"
    )
    warmup_slider = mo.ui.slider(
        start=0.0, stop=0.3, step=0.05, value=0.1, label="Warmup ratio"
    )
    save_dir_input = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Save directory",
    )
    load_dir_input = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Load directory",
    )
    run_btn = mo.ui.run_button(label="Run")
    controls = mo.vstack(
        [
            mode_radio,
            mo.hstack([lr_slider, batch_slider, epoch_slider]),
            mo.hstack([max_len_slider, warmup_slider]),
            mo.hstack([save_dir_input, load_dir_input]),
            run_btn,
        ]
    )
    controls
    return (
        batch_slider,
        epoch_slider,
        load_dir_input,
        lr_slider,
        max_len_slider,
        mode_radio,
        run_btn,
        save_dir_input,
        warmup_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    """)
    return


@app.cell
def _(Dataset, torch):
    class LyricsDataset(Dataset):
        def __init__(self, tokenizer, texts, labels, max_length):
            enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    return (LyricsDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LyricsDataset,
    Trainer,
    TrainingArguments,
    batch_slider,
    device,
    epoch_slider,
    f1_score,
    id2label,
    label2id,
    load_dir_input,
    lr_slider,
    max_len_slider,
    mo,
    mode_radio,
    np,
    run_btn,
    save_dir_input,
    test_labels,
    test_texts,
    train_labels,
    train_texts,
    warmup_slider,
):
    BERT_MODEL = "answerdotai/ModernBERT-base"

    if run_btn.value:
        mode = mode_radio.value
        max_length = max_len_slider.value
        n_epochs = epoch_slider.value
        batch_size = batch_slider.value
        lr = lr_slider.value * 1e-5
        warmup_ratio = warmup_slider.value
        save_dir = str(save_dir_input.path(0))
        model_source = (
            str(load_dir_input.path(0))
            if mode == "Continue training"
            else BERT_MODEL
        )

        mo.output.replace(mo.md("**Tokenizing dataset...**"))
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        train_ds = LyricsDataset(tokenizer, train_texts, train_labels, max_length)
        test_ds = LyricsDataset(tokenizer, test_texts, test_labels, max_length)

        mo.output.replace(mo.md(f"**Loading `{model_source}`...**"))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        def compute_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=-1)
            return {
                "f1": float(
                    f1_score(eval_pred.label_ids, preds, average="weighted")
                )
            }

        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            logging_steps=10,
            fp16=(device.type == "cuda"),
        )

        mo.output.replace(mo.md("**Training in progress...**"))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
        )
        trainer.train()
        model.config.training_max_length = max_length
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        log_history = trainer.state.log_history
        mo.output.replace(
            mo.md(f"**Training complete, model saved to `{save_dir}`.**")
        )
    else:
        log_history = []
        mo.output.replace(mo.md("*Adjust settings above and click **Run**.*"))
    return (log_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation

    Load any saved model to compute F1 scores on the full train and test splits.
    """)
    return


@app.cell
def _(mo):
    eval_source_radio = mo.ui.radio(
        options=["Local directory", "Hugging Face Hub"],
        value="Local directory",
        label="Load from",
    )
    eval_dir_input = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Model directory",
    )
    eval_hub_input = mo.ui.text(
        value="RichardSieg/",
        label="Hub model ID",
        placeholder="RichardSieg/model-name",
    )
    eval_run_btn = mo.ui.run_button(label="Evaluate")
    return eval_dir_input, eval_hub_input, eval_run_btn, eval_source_radio


@app.cell
def _(eval_dir_input, eval_hub_input, eval_run_btn, eval_source_radio, mo):
    eval_controls = mo.vstack(
        [
            eval_source_radio,
            eval_dir_input
            if eval_source_radio.value == "Local directory"
            else eval_hub_input,
            eval_run_btn,
        ]
    )
    eval_controls
    return


@app.cell
def _(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    classification_report,
    device,
    eval_dir_input,
    eval_hub_input,
    eval_run_btn,
    eval_source_radio,
    f1_score,
    id2label,
    mo,
    test_labels,
    test_texts,
    torch,
):
    if not eval_run_btn.value:
        bert_results = None
        _out = mo.md("*Select a model source and click **Evaluate**.*")
    else:
        if eval_source_radio.value == "Local directory":
            eval_dir = str(eval_dir_input.path(0))
        else:
            eval_dir = eval_hub_input.value.strip()

        tokenizer_pred = AutoTokenizer.from_pretrained(eval_dir)
        model_pred = AutoModelForSequenceClassification.from_pretrained(
            eval_dir
        ).to(device)
        eval_max_length = getattr(model_pred.config, "training_max_length", 512)
        model_pred.eval()

        def predict_texts(texts, batch_size=32):
            preds = []
            for i in range(0, len(texts), batch_size):
                enc = tokenizer_pred(
                    texts[i : i + batch_size],
                    truncation=True,
                    padding=True,
                    max_length=eval_max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    preds.extend(
                        model_pred(**enc).logits.argmax(dim=-1).cpu().tolist()
                    )
            return preds

        # train_preds = predict_texts(train_texts)
        test_preds = predict_texts(test_texts)

        # train_f1 = float(f1_score(train_labels, train_preds, average="weighted"))
        test_f1 = float(f1_score(test_labels, test_preds, average="weighted"))

        decade_names = [id2label[i] for i in sorted(id2label)]
        per_class_report = classification_report(
            test_labels, test_preds, target_names=decade_names, output_dict=True
        )
        report_str = classification_report(
            test_labels, test_preds, target_names=decade_names
        )

        bert_results = {
            # "train_f1": train_f1,
            "test_f1": test_f1,
            "per_class_report": per_class_report,
        }

        _out = mo.vstack(
            [
                mo.md(f" | **Test weighted F1: {test_f1:.4f}**"),
                mo.md(f"```\n{report_str}\n```"),
            ]
        )

    _out
    return (bert_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Curves
    """)
    return


@app.cell
def _(bert_results, go, log_history, mo):
    if not log_history or bert_results is None:
        _out = mo.md("*Complete training and evaluation to see results.*")
    else:
        step_logs = [
            e
            for e in log_history
            if "loss" in e and "eval_loss" not in e and "train_loss" not in e
        ]
        step_epochs = [e["epoch"] for e in step_logs]
        step_losses = [e["loss"] for e in step_logs]

        # Per-epoch eval metrics
        eval_logs = [e for e in log_history if "eval_loss" in e]
        eval_epochs = [e["epoch"] for e in eval_logs]
        eval_losses = [e["eval_loss"] for e in eval_logs]
        eval_f1s = [e.get("eval_f1", 0.0) for e in eval_logs]

        # Loss curves + eval F1 (dual y-axis)
        loss_fig = go.Figure()
        loss_fig.add_trace(
            go.Scatter(
                x=step_epochs,
                y=step_losses,
                name="Train loss",
                mode="lines",
                line=dict(color="#e76f51", width=1.5),
            )
        )
        loss_fig.add_trace(
            go.Scatter(
                x=eval_epochs,
                y=eval_losses,
                name="Eval loss",
                mode="lines+markers",
                line=dict(color="#2a9d8f", width=2.5),
                marker=dict(size=8),
            )
        )
        loss_fig.add_trace(
            go.Scatter(
                x=eval_epochs,
                y=eval_f1s,
                name="Eval weighted F1",
                mode="lines+markers",
                line=dict(color="#1f6fab", width=2.5),
                marker=dict(size=8),
                yaxis="y2",
            )
        )
        loss_fig.update_layout(
            title="Loss and Eval F1 per Epoch",
            xaxis_title="Epoch",
            yaxis=dict(title="Loss", side="left"),
            yaxis2=dict(
                title="Weighted F1",
                side="right",
                overlaying="y",
                range=[0, 1],
                showgrid=False,
            ),
            legend=dict(orientation="h", y=1.12),
            height=420,
            margin=dict(t=80, b=40),
            hovermode="x unified",
        )

        # Final train vs test F1
        f1_fig = go.Figure(
            go.Bar(
                x=["Train (final)", "Test (final)"],
                y=[bert_results["train_f1"], bert_results["test_f1"]],
                marker_color=["#f4a261", "#1f6fab"],
                text=[
                    f"{bert_results['train_f1']:.4f}",
                    f"{bert_results['test_f1']:.4f}",
                ],
                textposition="outside",
                textfont=dict(size=14),
                width=0.4,
            )
        )
        f1_fig.update_layout(
            title="Weighted F1 — Train vs Test",
            yaxis=dict(title="Weighted F1", range=[0, 1.1]),
            height=380,
            margin=dict(t=60, b=40),
        )

        _out = mo.vstack([loss_fig, f1_fig])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline: TF-IDF + Logistic Regression

    Each song is represented as a TF-IDF vector (10,000 terms, English stop words removed).
    Logistic Regression with balanced class weights compensates for slightly unequal decade sizes.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    TfidfVectorizer,
    classification_report,
    f1_score,
    id2label,
    mo,
    test_labels,
    test_texts,
    train_labels,
    train_texts,
):
    tfidf_vec = TfidfVectorizer(max_features=10000, min_df=2, stop_words="english")
    X_train_tfidf = tfidf_vec.fit_transform(train_texts)
    X_test_tfidf = tfidf_vec.transform(test_texts)

    logreg = LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, class_weight="balanced"
    )
    logreg.fit(X_train_tfidf, train_labels)

    tfidf_preds = logreg.predict(X_test_tfidf)
    tfidf_f1 = float(f1_score(test_labels, tfidf_preds, average="weighted"))

    decade_names_tfidf = [id2label[i] for i in sorted(id2label)]
    tfidf_report_dict = classification_report(
        test_labels, tfidf_preds, target_names=decade_names_tfidf, output_dict=True
    )
    tfidf_per_class_f1 = {
        d: tfidf_report_dict[d]["f1-score"] for d in decade_names_tfidf
    }
    tfidf_report_str = classification_report(
        test_labels, tfidf_preds, target_names=decade_names_tfidf
    )

    baseline_display = mo.vstack(
        [
            mo.md(f"**Weighted F1: {tfidf_f1:.4f}**"),
            mo.md(f"```\n{tfidf_report_str}\n```"),
        ]
    )
    baseline_display
    return tfidf_f1, tfidf_per_class_f1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison: F1 by Decade
    """)
    return


@app.cell
def _(bert_results, go, id2label, mo, tfidf_per_class_f1):
    if not bert_results:
        _out = mo.md("*Run training and prediction to see the comparison.*")
    else:
        decade_labels = [id2label[i] for i in sorted(id2label)]
        report = bert_results["per_class_report"]
        bert_f1s = [report.get(d, {}).get("f1-score", 0.0) for d in decade_labels]
        tfidf_f1s_list = [tfidf_per_class_f1.get(d, 0.0) for d in decade_labels]

        _fig = go.Figure(
            data=[
                go.Bar(
                    name="TF-IDF + LogReg",
                    x=decade_labels,
                    y=tfidf_f1s_list,
                    marker_color="#f4a261",
                ),
                go.Bar(
                    name="ModernBERT-base",
                    x=decade_labels,
                    y=bert_f1s,
                    marker_color="#1f6fab",
                ),
            ]
        )
        _fig.update_layout(
            title="F1 Score by Decade",
            xaxis_title="Decade",
            yaxis=dict(title="F1 Score", range=[0, 1.05]),
            barmode="group",
            height=420,
            margin=dict(t=60, b=40),
            legend=dict(orientation="h", y=1.12),
        )
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison: Overall Weighted F1
    """)
    return


@app.cell
def _(bert_results, go, mo, tfidf_f1):
    if not bert_results:
        _out = mo.md("*Run training and prediction to see the comparison.*")
    else:
        delta = bert_results["test_f1"] - tfidf_f1
        sign = "+" if delta >= 0 else ""
        _fig = go.Figure(
            go.Bar(
                x=["TF-IDF + LogReg", "ModernBERT-base"],
                y=[tfidf_f1, bert_results["test_f1"]],
                marker_color=["#f4a261", "#1f6fab"],
                text=[f"{tfidf_f1:.4f}", f"{bert_results['test_f1']:.4f}"],
                textposition="outside",
                textfont=dict(size=14),
                width=0.4,
            )
        )
        _fig.update_layout(
            title=f"Weighted F1 - ModernBERT vs Baseline ({sign}{delta:.4f})",
            yaxis=dict(title="Weighted F1", range=[0, 1.05]),
            height=400,
            margin=dict(t=60, b=40),
        )
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation: Word Attributions

    Select a song and a fine-tuned model to visualize which words drive the prediction.
    Attributions are computed with [transformers-interpret](https://github.com/cdpierse/transformers-interpret)
    (integrated gradients via Captum). The model runs on CPU for compatibility.
    """)
    return


@app.cell
def _(df, mo):
    get_interp_model, set_interp_model = mo.state(None)
    interp_source_radio = mo.ui.radio(
        options=["Local directory", "Hugging Face Hub"],
        value="Local directory",
        label="Load from",
    )
    interp_dir_input = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Model directory",
    )
    interp_hub_input = mo.ui.text(
        value="RichardSieg/",
        label="Hub model ID",
        placeholder="RichardSieg/model-name",
    )
    song_options = {}
    for _i, _row in df[["artist", "song_title"]].dropna().iterrows():
        _label = f"{_row['artist']} - {_row['song_title']}"
        if _label not in song_options:
            song_options[_label] = _i
    song_options = dict(sorted(song_options.items()))
    song_selector = mo.ui.dropdown(
        options=song_options, label="Artist - Song", searchable=True
    )
    load_model_btn = mo.ui.run_button(label="Load Model")
    explain_btn = mo.ui.run_button(label="Explain")
    return (
        explain_btn,
        get_interp_model,
        interp_dir_input,
        interp_hub_input,
        interp_source_radio,
        load_model_btn,
        set_interp_model,
        song_selector,
    )


@app.cell
def _(
    explain_btn,
    interp_dir_input,
    interp_hub_input,
    interp_source_radio,
    load_model_btn,
    mo,
    song_selector,
):
    interp_controls = mo.vstack(
        [
            interp_source_radio,
            interp_dir_input
            if interp_source_radio.value == "Local directory"
            else interp_hub_input,
            mo.hstack([load_model_btn, explain_btn]),
            song_selector,
        ]
    )
    interp_controls
    return


@app.cell
def _(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    interp_dir_input,
    interp_hub_input,
    interp_source_radio,
    load_model_btn,
    mo,
    set_interp_model,
):
    if load_model_btn.value:
        if interp_source_radio.value == "Local directory":
            _path = str(interp_dir_input.path(0))
        else:
            _path = interp_hub_input.value.strip()
        _tokenizer = AutoTokenizer.from_pretrained(_path)
        _model = AutoModelForSequenceClassification.from_pretrained(_path)
        _model.eval()
        set_interp_model({"model": _model, "tokenizer": _tokenizer, "path": _path})
        _out = mo.callout(mo.md(f"Model loaded from `{_path}`"), kind="success")
    else:
        _out = mo.md("*Select model source and click **Load Model**.*")
    _out
    return


@app.cell
def _(df, explain_btn, get_interp_model, mo, song_selector):
    if not explain_btn.value or song_selector.value is None:
        _out = mo.md("*Select a song and click **Explain**.*")
    else:
        _state = get_interp_model()
        if _state is None:
            _out = mo.callout(
                mo.md("**No model loaded.** Click **Load Model** first."),
                kind="warn",
            )
        else:
            from transformers_interpret import SequenceClassificationExplainer

            _row = df.loc[song_selector.value]
            _cls_explainer = SequenceClassificationExplainer(
                _state["model"], _state["tokenizer"]
            )
            _cls_explainer(_row["lyrics"])

            _cls_explainer.visualize("temp_interp_viz.html")
            with open("temp_interp_viz.html") as f:
                _viz_html = f.read()

            _out = mo.vstack(
                [
                    mo.md(
                        f"**{_row['artist']} - {_row['song_title']}** | "
                        f"**Predicted:** {_cls_explainer.predicted_class_name}"
                    ),
                    mo.Html(_viz_html),
                ]
            )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Push to Hugging Face Hub
    """)
    return


@app.cell
def _(mo):
    push_dir_input = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Model directory",
    )
    push_repo_input = mo.ui.text(
        value="RichardSieg/",
        label="Repository name",
        placeholder="RichardSieg/model-name",
    )
    push_btn = mo.ui.run_button(label="Push to Hub")
    push_controls = mo.vstack(
        [
            mo.hstack([push_dir_input, push_repo_input]),
            push_btn,
        ]
    )
    push_controls
    return push_btn, push_dir_input, push_repo_input


@app.cell
def _(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    mo,
    os,
    push_btn,
    push_dir_input,
    push_repo_input,
):
    if not push_btn.value:
        _out = mo.md(
            "*Select a saved model directory, enter a repository name, and click **Push to Hub**.*"
        )
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            _out = mo.callout(
                mo.md("**HF_TOKEN not found in environment.**"), kind="danger"
            )
        else:
            push_dir = str(push_dir_input.path(0))
            repo_id = push_repo_input.value.strip()

            push_tokenizer = AutoTokenizer.from_pretrained(push_dir)
            push_model = AutoModelForSequenceClassification.from_pretrained(
                push_dir
            )

            push_model.push_to_hub(repo_id, token=hf_token)
            push_tokenizer.push_to_hub(repo_id, token=hf_token)

            hub_url = f"https://huggingface.co/{repo_id}"
            _out = mo.callout(
                mo.md(f"Model pushed to [{repo_id}]({hub_url})"), kind="success"
            )
    _out
    return


if __name__ == "__main__":
    app.run()
