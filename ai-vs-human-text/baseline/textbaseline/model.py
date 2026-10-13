"""The baseline: TF-IDF word 1-2 grams and logistic regression.

Same settings as the first cross-validation run in 1-Log-TFIDF.ipynb, before
Optuna tuning. Kept fixed on purpose: no search, no ensembling.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42

VECTORIZER_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}
CLASSIFIER_PARAMS = {
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": SEED,
}


def build_pipeline(min_df: int | None = None) -> Pipeline:
    """A fresh, unfitted pipeline. The vectorizer is fitted with the classifier,
    so it only ever sees the rows passed to fit()."""
    params = dict(VECTORIZER_PARAMS)
    if min_df is not None:
        params["min_df"] = min_df
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**params)),
            ("clf", LogisticRegression(**CLASSIFIER_PARAMS)),
        ]
    )


def describe() -> dict:
    return {
        "name": "tfidf-word12-logreg",
        "vectorizer": {**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])},
        "classifier": {"type": "LogisticRegression", **CLASSIFIER_PARAMS},
        "fit_scope": "vectorizer and classifier are fitted together on the training folds only",
    }
