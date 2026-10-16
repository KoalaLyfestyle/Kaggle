"""Fit the baseline on all unique training texts and score the test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from . import data, model
from .evaluate import environment


def train(data_dir: Path, out_dir: Path, min_df: int | None = None) -> dict:
    train_data = data.load_train(data_dir)
    pipe = model.build_pipeline(min_df=min_df)
    pipe.fit(train_data.frame["text"], train_data.frame["label"])

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    info = {
        "model": model.describe(),
        "data": train_data.checks,
        "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_),
        "environment": environment(),
    }
    test = data.load_test(data_dir)
    if test is not None:
        test["probability"] = pipe.predict_proba(test["text"])[:, 1]
        test[["id", "probability"]].to_csv(out_dir / "submission.csv", index=False)
        info["submission"] = data.display_path(out_dir / "submission.csv")
        info["test_rows"] = int(len(test))
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
