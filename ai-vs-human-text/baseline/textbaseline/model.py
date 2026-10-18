"""The baseline: TF-IDF word 1-2 grams and logistic regression.

Same settings as the first cross-validation run in 1-Log-TFIDF.ipynb, before
Optuna tuning. Kept fixed on purpose: no search, no ensembling.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


class TopTermsTfidf(BaseEstimator, TransformerMixin):
    """TF-IDF over the max_features most frequent terms, with a fixed tie-break.

    TfidfVectorizer(max_features=...) breaks ties in term frequency with an
    unstable sort, so the kept vocabulary, and every metric after it, can
    differ between CPU architectures. Here ties break alphabetically: terms
    are ranked by total count (highest first), then by name. Fitted inside
    the pipeline, so it only sees the training rows.
    """

    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        counter = CountVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df)
        counts = counter.fit_transform(X)
        names = counter.get_feature_names_out()  # alphabetical
        totals = np.asarray(counts.sum(axis=0)).ravel()
        # lexsort uses the last key first: count descending, then name.
        order = np.lexsort((np.arange(len(names)), -totals))[: self.max_features]
        self.kept_terms_ = sorted(names[order].tolist())
        self.vectorizer_ = TfidfVectorizer(
            vocabulary=self.kept_terms_, ngram_range=self.ngram_range, sublinear_tf=self.sublinear_tf
        ).fit(X)
        self.vocabulary_ = self.vectorizer_.vocabulary_
        return self

    def transform(self, X):
        return self.vectorizer_.transform(X)


def build_pipeline(min_df: int | None = None) -> Pipeline:
    """A fresh, unfitted pipeline. The vectorizer is fitted with the classifier,
    so it only ever sees the rows passed to fit()."""
    params = dict(VECTORIZER_PARAMS)
    if min_df is not None:
        params["min_df"] = min_df
    return Pipeline(
        [
            ("tfidf", TopTermsTfidf(**params)),
            ("clf", LogisticRegression(**CLASSIFIER_PARAMS)),
        ]
    )


def describe() -> dict:
    return {
        "name": "tfidf-word12-logreg",
        "vectorizer": {**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])},
        "classifier": {"type": "LogisticRegression", **CLASSIFIER_PARAMS},
        "vocabulary_tie_break": "terms ranked by total count, ties broken alphabetically",
        "fit_scope": "vectorizer and classifier are fitted together on the training folds only",
    }
