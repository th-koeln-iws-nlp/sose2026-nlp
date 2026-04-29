import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        precision_recall_curve,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MinMaxScaler
    import pymde

    return (
        LogisticRegression,
        MinMaxScaler,
        MultinomialNB,
        TfidfVectorizer,
        average_precision_score,
        classification_report,
        go,
        make_subplots,
        mo,
        np,
        pd,
        precision_recall_curve,
        precision_score,
        px,
        pymde,
        recall_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Text Classification on Song Lyrics

    We represent each song as a TF-IDF vector and compare two classifiers:

    - **Multinomial Naive Bayes**: a fast, count-based probabilistic baseline
    - **Logistic Regression**: a discriminative model that learns per-term weights per class

    The notebook covers two tasks of increasing difficulty:

    1. **Explicit content detection** (binary) -- the label is directly encoded in the words
    2. **Decade prediction** (6-class) -- a harder task where the signal is more subtle
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1: Explicit Content Detection

    Can the model tell whether a song has explicit lyrics? The Spotify `explicit` flag is our label. Because profanity and explicit vocabulary are highly distinctive TF-IDF features, this should be much easier than predicting decade or genre.

    About half the dataset has a Spotify match and therefore an explicit label. Of those, roughly 10% are marked explicit, so the classes are heavily skewed.
    """)
    return


@app.cell
def _(pd):
    df_exp_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df_exp = df_exp_raw[df_exp_raw["explicit"].notna()].copy()
    df_exp["explicit_label"] = df_exp["explicit"].map(
        {True: "Explicit", False: "Not Explicit"}
    )
    exp_counts = df_exp["explicit_label"].value_counts()
    print(
        f"{len(df_exp)} songs | Explicit: {exp_counts['Explicit']} ({exp_counts['Explicit'] / len(df_exp) * 100:.1f}%) | Not Explicit: {exp_counts['Not Explicit']}"
    )
    return (df_exp,)


@app.cell
def _(df_exp, px):
    exp_counts_sorted = df_exp["explicit_label"].value_counts().sort_index()
    fig_exp_dist = px.bar(
        x=exp_counts_sorted.index,
        y=exp_counts_sorted.values,
        labels={"x": "", "y": "Number of Songs"},
        title="Class Distribution: Explicit vs. Not Explicit",
        text=exp_counts_sorted.values,
    )
    fig_exp_dist.update_traces(textposition="outside")
    fig_exp_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A naive classifier that always predicts "Not Explicit" would reach ~90% accuracy, so accuracy alone is misleading. We focus on precision, recall, and F1 for the minority class.

    To counteract the imbalance, both models are adjusted: Logistic Regression uses `class_weight="balanced"`, and Multinomial NB uses `class_prior=[0.5, 0.5]` (uniform prior), which is the NB equivalent of ignoring the majority-class prior.
    """)
    return


@app.cell
def _(TfidfVectorizer, df_exp, np, train_test_split):
    vectorizer_exp = TfidfVectorizer(
        max_features=10000, min_df=2, stop_words="english"
    )
    X_exp = vectorizer_exp.fit_transform(df_exp["lyrics"])
    y_exp = df_exp["explicit_label"].values

    train_idx_exp, test_idx_exp = train_test_split(
        np.arange(len(df_exp)), test_size=0.2, random_state=42, stratify=y_exp
    )
    X_train_exp = X_exp[train_idx_exp]
    X_test_exp = X_exp[test_idx_exp]
    y_train_exp = y_exp[train_idx_exp]
    y_test_exp = y_exp[test_idx_exp]
    print(
        f"Train: {X_train_exp.shape[0]} | Test: {X_test_exp.shape[0]} | Vocab: {X_train_exp.shape[1]:,} terms"
    )
    return (
        X_test_exp,
        X_train_exp,
        test_idx_exp,
        vectorizer_exp,
        y_test_exp,
        y_train_exp,
    )


@app.cell
def _(LogisticRegression, MultinomialNB, X_train_exp, y_train_exp):
    nb_exp = MultinomialNB(class_prior=[0.5, 0.5])
    nb_exp.fit(X_train_exp, y_train_exp)

    logreg_exp = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    logreg_exp.fit(X_train_exp, y_train_exp)
    return logreg_exp, nb_exp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification Reports
    """)
    return


@app.cell
def _(X_test_exp, classification_report, logreg_exp, mo, nb_exp, y_test_exp):
    nb_exp_report = classification_report(y_test_exp, nb_exp.predict(X_test_exp))
    logreg_exp_report = classification_report(
        y_test_exp, logreg_exp.predict(X_test_exp)
    )
    mo.vstack(
        [
            mo.md(f"### Multinomial Naive Bayes\n```\n{nb_exp_report}\n```"),
            mo.md(
                f"### Logistic Regression (balanced)\n```\n{logreg_exp_report}\n```"
            ),
        ]
    )
    return


@app.cell
def _(X_test_exp, df_exp, logreg_exp, mo, np, test_idx_exp, y_test_exp):
    proba_exp = logreg_exp.predict_proba(X_test_exp)
    exp_col = list(logreg_exp.classes_).index("Explicit")
    prob_explicit = proba_exp[:, exp_col]
    y_pred_exp = logreg_exp.predict(X_test_exp)

    fp_pos = np.where((y_pred_exp == "Explicit") & (y_test_exp == "Not Explicit"))[
        0
    ]
    fp_top2 = fp_pos[prob_explicit[fp_pos].argsort()[::-1][:3]]

    fn_pos = np.where((y_pred_exp == "Not Explicit") & (y_test_exp == "Explicit"))[
        0
    ]
    fn_top2 = fn_pos[prob_explicit[fn_pos].argsort()[:3]]


    def error_table(positions):
        rows = df_exp.iloc[test_idx_exp[positions]][
            ["artist", "song_title", "year", "lyrics"]
        ].copy()
        rows["p(Explicit)"] = prob_explicit[positions].round(3)
        return rows.reset_index(drop=True)


    mo.vstack(
        [
            mo.md("### Most Extreme Errors (Logistic Regression)"),
            mo.md(
                "**False Positives** -- predicted Explicit, actually Not Explicit"
            ),
            mo.ui.table(error_table(fp_top2)),
            mo.md(
                "**False Negatives** -- predicted Not Explicit, actually Explicit"
            ),
            mo.ui.table(error_table(fn_top2)),
        ]
    )
    return


@app.cell
def _(
    X_test_exp,
    average_precision_score,
    go,
    logreg_exp,
    np,
    precision_recall_curve,
    precision_score,
    recall_score,
    y_test_exp,
):
    y_true_bin = (y_test_exp == "Explicit").astype(int)
    prob_exp_pr = logreg_exp.predict_proba(X_test_exp)[
        :, list(logreg_exp.classes_).index("Explicit")
    ]

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_bin, prob_exp_pr)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    pr_thresholds_padded = np.append(pr_thresholds, np.nan)
    ap = average_precision_score(y_true_bin, prob_exp_pr)

    y_pred_bin = (logreg_exp.predict(X_test_exp) == "Explicit").astype(int)
    op_precision = precision_score(y_true_bin, y_pred_bin)
    op_recall = recall_score(y_true_bin, y_pred_bin)

    baseline = y_true_bin.mean()

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode="lines",
        name=f"Random baseline ({baseline:.2f})",
        line={"color": "grey", "width": 1, "dash": "dash"},
    ))
    fig_pr.add_trace(go.Scatter(
        x=pr_recall, y=pr_precision,
        mode="lines",
        name="LogReg",
        line={"color": "steelblue", "width": 2},
        fill="tozeroy",
        fillcolor="rgba(70, 130, 180, 0.12)",
        customdata=pr_thresholds_padded,
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>t = %{customdata:.3f}<extra></extra>",
    ))
    fig_pr.add_trace(go.Scatter(
        x=[op_recall], y=[op_precision],
        mode="markers",
        name="Default threshold (p=0.5)",
        marker={"color": "crimson", "size": 10, "symbol": "diamond"},
    ))
    fig_pr.update_layout(
        title="Precision-Recall Curve: Explicit Content",
        xaxis={"title": "Recall", "range": [0, 1]},
        yaxis={"title": "Precision", "range": [0, 1]},
    )
    fig_pr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Importance: Most Distinctive Words

    Since this is binary classification there is no dropdown. We show the top 10 words driving the "Explicit" prediction for each model.

    - **Logistic Regression**: top positive coefficients (words that increase the log-odds of predicting Explicit)
    - **Multinomial NB**: top log-likelihood ratio between the Explicit and Not Explicit classes
    """)
    return


@app.cell
def _(go, logreg_exp, make_subplots, nb_exp, vectorizer_exp):
    terms_exp = vectorizer_exp.get_feature_names_out()

    # For binary LogReg, coef_[0] positive values predict classes_[1] = "Not Explicit"
    # (sklearn sorts classes alphabetically: ["Explicit", "Not Explicit"])
    # So the most negative values are the strongest predictors for "Explicit"
    lr_exp_coefs = logreg_exp.coef_[0]
    lr_exp_top = lr_exp_coefs.argsort()[:10]
    lr_exp_display = -lr_exp_coefs[lr_exp_top]

    exp_cls_idx = list(nb_exp.classes_).index("Explicit")
    notexp_cls_idx = list(nb_exp.classes_).index("Not Explicit")
    nb_exp_ratio = (
        nb_exp.feature_log_prob_[exp_cls_idx]
        - nb_exp.feature_log_prob_[notexp_cls_idx]
    )
    nb_exp_top = nb_exp_ratio.argsort()[::-1][:10]

    fig_exp_feat = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Logistic Regression (coefficient)",
            "Multinomial NB (log-likelihood ratio)",
        ],
        horizontal_spacing=0.18,
    )
    fig_exp_feat.add_trace(
        go.Bar(
            x=lr_exp_display[::-1],
            y=terms_exp[lr_exp_top][::-1],
            orientation="h",
            marker_color="steelblue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_exp_feat.add_trace(
        go.Bar(
            x=nb_exp_ratio[nb_exp_top][::-1],
            y=terms_exp[nb_exp_top][::-1],
            orientation="h",
            marker_color="darkorange",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_exp_feat.update_layout(
        title="Top 10 Words Driving 'Explicit' Predictions",
        height=420,
    )
    fig_exp_feat.update_xaxes(title_text="Coefficient", col=1)
    fig_exp_feat.update_xaxes(title_text="Log-Likelihood Ratio", col=2)
    fig_exp_feat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decision Boundaries
    """)
    return


@app.cell(hide_code=True)
def _(MinMaxScaler, X_test_exp, X_train_exp, pymde):
    from scipy.sparse import vstack as _vstack
    X_combined_exp = _vstack([X_train_exp, X_test_exp])
    mde_exp = pymde.preserve_neighbors(X_combined_exp, embedding_dim=2, verbose=False)
    embedding_exp = mde_exp.embed(verbose=False).numpy()
    n_train_exp = X_train_exp.shape[0]
    coords_train_exp = embedding_exp[:n_train_exp]
    coords_test_exp = embedding_exp[n_train_exp:]

    scaler_exp = MinMaxScaler()
    coords_train_exp_scaled = scaler_exp.fit_transform(coords_train_exp)
    coords_test_exp_scaled = scaler_exp.transform(coords_test_exp)
    return coords_test_exp_scaled, coords_train_exp_scaled


@app.cell(hide_code=True)
def _(LogisticRegression, MultinomialNB, coords_train_exp_scaled, y_train_exp):
    nb_2d_exp = MultinomialNB(class_prior=[0.5, 0.5])
    nb_2d_exp.fit(coords_train_exp_scaled, y_train_exp)

    logreg_2d_exp = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    logreg_2d_exp.fit(coords_train_exp_scaled, y_train_exp)
    return logreg_2d_exp, nb_2d_exp


@app.cell(hide_code=True)
def _(
    coords_test_exp_scaled,
    go,
    logreg_2d_exp,
    make_subplots,
    nb_2d_exp,
    np,
    px,
    y_test_exp,
):
    labels_exp = sorted(set(y_test_exp))
    label_to_num_exp = {l: i for i, l in enumerate(labels_exp)}
    palette_exp = px.colors.qualitative.Plotly[:2]
    n_exp = 2

    colorscale_exp = []
    for _i, _c in enumerate(palette_exp):
        colorscale_exp.extend([[_i / n_exp, _c], [(_i + 1) / n_exp, _c]])

    res_exp = 300
    xs_exp = np.linspace(0, 1, res_exp)
    ys_exp = np.linspace(0, 1, res_exp)
    xx_exp, yy_exp = np.meshgrid(xs_exp, ys_exp)
    grid_exp = np.c_[xx_exp.ravel(), yy_exp.ravel()]

    Z_nb_exp = np.array(
        [label_to_num_exp[l] for l in nb_2d_exp.predict(grid_exp)]
    ).reshape(res_exp, res_exp)
    Z_lr_exp = np.array(
        [label_to_num_exp[l] for l in logreg_2d_exp.predict(grid_exp)]
    ).reshape(res_exp, res_exp)

    fig_exp_boundaries = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Multinomial Naive Bayes",
            "Logistic Regression (balanced)",
        ],
        horizontal_spacing=0.08,
    )
    for _col, _Z in enumerate([Z_nb_exp, Z_lr_exp], start=1):
        fig_exp_boundaries.add_trace(
            go.Heatmap(
                x=xs_exp,
                y=ys_exp,
                z=_Z,
                colorscale=colorscale_exp,
                zmin=-0.5,
                zmax=n_exp - 0.5,
                showscale=False,
                opacity=0.4,
            ),
            row=1,
            col=_col,
        )
        for _i, _label in enumerate(labels_exp):
            _mask = y_test_exp == _label
            fig_exp_boundaries.add_trace(
                go.Scatter(
                    x=coords_test_exp_scaled[_mask, 0],
                    y=coords_test_exp_scaled[_mask, 1],
                    mode="markers",
                    marker={
                        "color": palette_exp[_i],
                        "size": 5,
                        "line": {"width": 0.5, "color": "white"},
                    },
                    name=_label,
                    showlegend=(_col == 1),
                    legendgroup=_label,
                ),
                row=1,
                col=_col,
            )
    fig_exp_boundaries.update_layout(
        title="Decision Boundaries: Explicit vs. Not Explicit (2D PyMDE Projection)",
        height=520,
        legend_title="Label",
    )
    fig_exp_boundaries.update_xaxes(showgrid=False, title_text="PyMDE 1")
    fig_exp_boundaries.update_yaxes(showgrid=False, title_text="PyMDE 2")
    fig_exp_boundaries
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2: Decade Prediction

    A harder task: can the model tell which decade a song is from? Songs before 1960 and from 2020 onward are excluded, leaving six decades (1960s to 2010s).
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    df = df_raw[(df_raw["year"] >= 1960) & (df_raw["year"] <= 2019)].copy()
    df["decade"] = df["year"].apply(lambda y: f"{(y // 10) * 10}s")
    print(f"{len(df)} songs | decades: {sorted(df['decade'].unique())}")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Distribution
    """)
    return


@app.cell
def _(df, px):
    decade_counts = df["decade"].value_counts().sort_index()
    fig_dist = px.bar(
        x=decade_counts.index,
        y=decade_counts.values,
        labels={"x": "Decade", "y": "Number of Songs"},
        title="Songs per Decade",
        text=decade_counts.values,
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TF-IDF Vectorization

    Each song is represented as a TF-IDF vector: up to 10,000 terms, English stop words removed, minimum document frequency of 2. We split 80/20 into train and test, stratified by decade so each class is proportionally represented.
    """)
    return


@app.cell
def _(TfidfVectorizer, df, train_test_split):
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=2, stop_words="english"
    )
    X_all = vectorizer.fit_transform(df["lyrics"])
    y_all = df["decade"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(
        f"Train: {X_train.shape[0]} songs | Test: {X_test.shape[0]} songs | Vocab: {X_train.shape[1]:,} terms"
    )
    return X_test, X_train, vectorizer, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(LogisticRegression, MultinomialNB, X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    logreg = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    logreg.fit(X_train, y_train)
    return logreg, nb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification Reports
    """)
    return


@app.cell
def _(X_test, classification_report, logreg, mo, nb, y_test):
    nb_report = classification_report(y_test, nb.predict(X_test))
    logreg_report = classification_report(y_test, logreg.predict(X_test))
    mo.vstack(
        [
            mo.md(f"### Multinomial Naive Bayes\n```\n{nb_report}\n```"),
            mo.md(f"### Logistic Regression\n```\n{logreg_report}\n```"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Importance: Most Distinctive Words per Decade

    Select a decade to see which words drive each classifier's decision.

    - **Logistic Regression**: the top 10 terms by coefficient value for that class. A higher coefficient means the word pushes the model more strongly toward predicting this decade.
    - **Multinomial Naive Bayes**: the top 10 terms by log-likelihood ratio, i.e., how much more probable a word is in this decade compared to the corpus average. This controls for overall word frequency and highlights what is truly characteristic.
    """)
    return


@app.cell
def _(logreg, mo):
    decade_dropdown = mo.ui.dropdown(
        options=list(logreg.classes_),
        value=list(logreg.classes_)[0],
        label="Select decade",
    )
    decade_dropdown
    return (decade_dropdown,)


@app.cell(hide_code=True)
def _(decade_dropdown, go, logreg, make_subplots, nb, vectorizer):
    terms = vectorizer.get_feature_names_out()
    selected_decade = decade_dropdown.value

    lr_class_idx = list(logreg.classes_).index(selected_decade)
    lr_coefs = logreg.coef_[lr_class_idx]
    lr_top_idx = lr_coefs.argsort()[::-1][:10]

    nb_class_idx = list(nb.classes_).index(selected_decade)
    nb_log_ratio = nb.feature_log_prob_[nb_class_idx] - nb.feature_log_prob_.mean(
        axis=0
    )
    nb_top_idx = nb_log_ratio.argsort()[::-1][:10]

    fig_feat = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Logistic Regression (coefficient)",
            "Multinomial NB (log-likelihood ratio)",
        ],
        horizontal_spacing=0.18,
    )
    fig_feat.add_trace(
        go.Bar(
            x=lr_coefs[lr_top_idx][::-1],
            y=terms[lr_top_idx][::-1],
            orientation="h",
            marker_color="steelblue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_feat.add_trace(
        go.Bar(
            x=nb_log_ratio[nb_top_idx][::-1],
            y=terms[nb_top_idx][::-1],
            orientation="h",
            marker_color="darkorange",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_feat.update_layout(
        title=f"Top 10 Distinctive Words: {selected_decade}",
        height=420,
    )
    fig_feat.update_xaxes(title_text="Coefficient", col=1)
    fig_feat.update_xaxes(title_text="Log-Likelihood Ratio", col=2)
    fig_feat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Boundaries

    TF-IDF vectors span thousands of dimensions. To visualize decision boundaries we project the data to 2D with PyMDE and retrain both classifiers in that reduced space. PyMDE coordinates are scaled to [0, 1] so Multinomial NB (which requires non-negative features) can operate on them.

    The colored background shows each model's predicted decade for every point on the 2D plane. Actual test-set songs are plotted on top with the same color scheme.

    This is a 2D approximation for visual insight. The classification reports above reflect full-dimensional performance.
    """)
    return


@app.cell(hide_code=True)
def _(MinMaxScaler, X_test, X_train, pymde):
    from scipy.sparse import vstack as _vstack
    X_combined = _vstack([X_train, X_test])
    mde = pymde.preserve_neighbors(X_combined, embedding_dim=2, verbose=False)
    embedding_all = mde.embed(verbose=False).numpy()
    n_train = X_train.shape[0]
    coords_train = embedding_all[:n_train]
    coords_test = embedding_all[n_train:]

    scaler = MinMaxScaler()
    coords_train_scaled = scaler.fit_transform(coords_train)
    coords_test_scaled = scaler.transform(coords_test)
    return coords_test_scaled, coords_train_scaled


@app.cell(hide_code=True)
def _(LogisticRegression, MultinomialNB, coords_train_scaled, y_train):
    nb_2d = MultinomialNB()
    nb_2d.fit(coords_train_scaled, y_train)

    logreg_2d = LogisticRegression(max_iter=1000, random_state=42)
    logreg_2d.fit(coords_train_scaled, y_train)
    return logreg_2d, nb_2d


@app.cell(hide_code=True)
def _(coords_test_scaled, go, logreg_2d, make_subplots, nb_2d, np, px, y_test):
    decades_sorted = sorted(set(y_test))
    decade_to_num = {d: i for i, d in enumerate(decades_sorted)}
    palette = px.colors.qualitative.Plotly[: len(decades_sorted)]
    n_decades = len(decades_sorted)

    # Step colorscale: each integer class value maps to one discrete color band
    colorscale = []
    for _i, _c in enumerate(palette):
        colorscale.extend([[_i / n_decades, _c], [(_i + 1) / n_decades, _c]])

    res = 300
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_nb = np.array([decade_to_num[d] for d in nb_2d.predict(grid)]).reshape(
        res, res
    )
    Z_lr = np.array([decade_to_num[d] for d in logreg_2d.predict(grid)]).reshape(
        res, res
    )

    fig_boundaries = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Multinomial Naive Bayes", "Logistic Regression"],
        horizontal_spacing=0.08,
    )

    for _col, _Z in enumerate([Z_nb, Z_lr], start=1):
        fig_boundaries.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=_Z,
                colorscale=colorscale,
                zmin=-0.5,
                zmax=n_decades - 0.5,
                showscale=False,
                opacity=0.4,
            ),
            row=1,
            col=_col,
        )

        for _i, _decade in enumerate(decades_sorted):
            _mask = y_test == _decade
            fig_boundaries.add_trace(
                go.Scatter(
                    x=coords_test_scaled[_mask, 0],
                    y=coords_test_scaled[_mask, 1],
                    mode="markers",
                    marker={
                        "color": palette[_i],
                        "size": 5,
                        "line": {"width": 0.5, "color": "white"},
                    },
                    name=_decade,
                    showlegend=(_col == 1),
                    legendgroup=_decade,
                ),
                row=1,
                col=_col,
            )

    fig_boundaries.update_layout(
        title="Decision Boundaries on 2D PyMDE Projection of TF-IDF Vectors",
        height=520,
        legend_title="Decade",
    )
    fig_boundaries.update_xaxes(showgrid=False, title_text="PyMDE 1")
    fig_boundaries.update_yaxes(showgrid=False, title_text="PyMDE 2")
    fig_boundaries
    return


if __name__ == "__main__":
    app.run()
