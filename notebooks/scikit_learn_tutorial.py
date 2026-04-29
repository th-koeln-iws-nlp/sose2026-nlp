import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px

    return mo, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genre Classification with Scikit-learn

    We use Spotify audio features from the Billboard Top 100 dataset to classify songs by genre using logistic regression.

    The audio features are numeric descriptors extracted by Spotify (danceability, energy, loudness, etc.). Note that a large share of songs are missing either audio features or genre labels, so we filter to complete rows before training.
    """)
    return


@app.cell
def _(pd):
    billboard_df = pd.read_csv("data/billboard_top100/billboard_top_100.csv")
    return (billboard_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the Data

    We keep only rows that have both audio features and a genre label. The nine Spotify audio features used as input are: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo.
    """)
    return


@app.cell
def _(billboard_df):
    AUDIO_FEATURES = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ]

    songs_clean = billboard_df.dropna(subset=AUDIO_FEATURES + ["genre"])
    print(
        f"{len(songs_clean)} of {len(billboard_df)} songs have complete audio features and a genre label."
    )
    return AUDIO_FEATURES, songs_clean


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Genre Distribution
    """)
    return


@app.cell
def _(px, songs_clean):
    genre_counts = songs_clean["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]

    fig_genre = px.bar(
        genre_counts,
        x="genre",
        y="count",
        title=f"Genre distribution in filtered dataset (n={len(songs_clean)})",
        labels={"genre": "Genre", "count": "Number of songs"},
    )
    fig_genre
    return


@app.cell
def _(songs_clean):
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        songs_clean, test_size=0.2, random_state=42
    )
    return test_df, train_df


@app.cell
def _(AUDIO_FEATURES, train_df):
    X_train = train_df[AUDIO_FEATURES].to_numpy()
    y_train = train_df["genre"].to_numpy()
    print("Training set shape:", X_train.shape)
    return X_train, y_train


@app.cell
def _(AUDIO_FEATURES, test_df):
    X_test = test_df[AUDIO_FEATURES].to_numpy()
    y_test = test_df["genre"].to_numpy()
    return X_test, y_test


@app.cell
def _(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    return LogisticRegression, clf


@app.cell
def _(X_test, clf, y_test):
    from sklearn.metrics import classification_report

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return (classification_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Logistic Regression Coefficients

    Each row shows how strongly a feature pushes the model toward that genre. Positive values increase the probability, negative values decrease it.
    """)
    return


@app.cell
def _(AUDIO_FEATURES, clf, pd, px):
    coef_df = pd.DataFrame(clf.coef_, columns=AUDIO_FEATURES, index=clf.classes_)
    fig_coef = px.imshow(
        coef_df,
        title="Logistic regression coefficients by genre",
        labels={"x": "Feature", "y": "Genre", "color": "Coefficient"},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto",
    )
    fig_coef
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature Scaling

    KNN and other distance-based classifiers are sensitive to feature scale. `StandardScaler` standardizes each feature to mean 0 and standard deviation 1. We fit the scaler on training data only and apply the same transformation to the test set.
    """)
    return


@app.cell
def _(X_test, X_train):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return StandardScaler, X_test_scaled, X_train_scaled


@app.cell
def _(AUDIO_FEATURES, X_train, X_train_scaled, pd, px):
    df_original = pd.DataFrame(X_train, columns=AUDIO_FEATURES)
    df_scaled = pd.DataFrame(X_train_scaled, columns=AUDIO_FEATURES)

    stats_original = df_original.agg(["mean", "std"]).T.reset_index()
    stats_original.columns = ["feature", "mean", "std"]
    stats_original["version"] = "original"

    stats_scaled = df_scaled.agg(["mean", "std"]).T.reset_index()
    stats_scaled.columns = ["feature", "mean", "std"]
    stats_scaled["version"] = "scaled"

    comparison_long = pd.concat([stats_original, stats_scaled]).melt(
        id_vars=["feature", "version"], var_name="stat", value_name="value"
    )

    fig_scale = px.bar(
        comparison_long,
        x="feature",
        y="value",
        color="version",
        facet_row="stat",
        barmode="group",
        title="Feature statistics before and after StandardScaler",
        labels={"value": "Value", "feature": "Feature"},
    )
    fig_scale
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Logistic Regression with Pipeline

    A `Pipeline` chains preprocessing and model into a single object. It fits the scaler on training data and applies the same transformation to the test data automatically, avoiding data leakage.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    X_test,
    X_train,
    classification_report,
    y_test,
    y_train,
):
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10000)),
    ])
    pipe.fit(X_train, y_train)
    y_pred_pipe = pipe.predict(X_test)
    print(classification_report(y_test, y_pred_pipe))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hyperparameter Tuning with GridSearchCV
    """)
    return


@app.cell
def _(LogisticRegression, X_train_scaled, y_train):
    from sklearn.model_selection import GridSearchCV

    param_grid = [{"C": [0.001, 0.01, 1, 10, 100, 1000]}]
    grid = GridSearchCV(
        LogisticRegression(max_iter=10000), param_grid, cv=5, scoring="accuracy"
    )
    grid.fit(X_train_scaled, y_train)
    print("Best C:", grid.best_params_["C"])
    return (grid,)


@app.cell
def _(X_test_scaled, classification_report, grid, y_test):
    y_pred_grid = grid.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_grid))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## K-Nearest Neighbors

    KNN classifies each song by majority vote among its k nearest neighbors in feature space. Because it relies on distances, it requires scaled features.
    """)
    return


@app.cell
def _(X_test_scaled, X_train_scaled, classification_report, y_test, y_train):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_knn))
    return


if __name__ == "__main__":
    app.run()
