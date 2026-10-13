"""Cross-validate the baseline on fixed folds and write a structured report."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from . import data, model

REPORT_SCHEMA = "textbaseline.report/v1"
METRICS = ("roc_auc", "log_loss", "accuracy", "f1", "brier")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def evaluate(
    data_dir: Path,
    splits_path: Path,
    out_dir: Path,
    min_df: int | None = None,
    label: str = "competition",
    n_errors: int = 10,
) -> dict:
    train = data.load_train(data_dir)
    test = data.load_test(data_dir)
    frame = train.frame.copy()
    frame["fold"] = data.read_folds(splits_path, frame)

    folds = sorted(frame["fold"].unique())
    oof = np.full(len(frame), np.nan)
    per_fold = []
    shared_texts = 0
    for fold in folds:
        train_rows = frame["fold"] != fold
        valid_rows = ~train_rows
        shared_texts += len(set(frame.loc[train_rows, "text_id"]) & set(frame.loc[valid_rows, "text_id"]))
        # Train-only fitting: a new pipeline, fitted on the training folds alone.
        pipe = model.build_pipeline(min_df=min_df)
        pipe.fit(frame.loc[train_rows, "text"], frame.loc[train_rows, "label"])
        p = pipe.predict_proba(frame.loc[valid_rows, "text"])[:, 1]
        oof[valid_rows.to_numpy()] = p
        per_fold.append(
            {
                "fold": int(fold),
                "train_rows": int(train_rows.sum()),
                "valid_rows": int(valid_rows.sum()),
                **_metrics(frame.loc[valid_rows, "label"].to_numpy(), p),
            }
        )

    y = frame["label"].to_numpy()
    summary = {
        name: {
            "mean": float(np.mean([f[name] for f in per_fold])),
            "std": float(np.std([f[name] for f in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0,
        }
        for name in METRICS
    }
    pooled = _metrics(y, oof)

    frame["prob_ai"] = oof
    frame["error"] = np.abs(frame["label"] - frame["prob_ai"])
    wrong = frame[(frame["prob_ai"] >= 0.5) != (frame["label"] == 1)]
    # The most confident mistakes of each kind, so both failure modes show.
    worst = pd.concat(
        [
            wrong[wrong["label"] == 0].sort_values("error", ascending=False).head(n_errors // 2),
            wrong[wrong["label"] == 1].sort_values("error", ascending=False).head(n_errors - n_errors // 2),
        ]
    )
    examples = [
        {
            "text_id": r.text_id,
            "fold": int(r.fold),
            "label": int(r.label),
            "prob_ai": round(float(r.prob_ai), 4),
            "kind": "false positive (human text scored as AI)" if r.label == 0 else "false negative (AI text scored as human)",
            "snippet": " ".join(r.text.split())[:240],
        }
        for r in worst.itertuples()
    ]

    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": label,
        "reproduces_competition": label == "competition",
        "data": {**train.checks, **data.overlap_checks(train, test)},
        "split": {
            "method": "stratified 5-fold over unique texts (exact and whitespace/case duplicates merged)",
            "manifest": str(splits_path),
            "manifest_sha256": data.sha256_file(splits_path),
            "folds": len(folds),
            "validation_texts_also_in_training_folds": int(shared_texts),
        },
        "model": model.describe() | ({"vectorizer_min_df_override": min_df} if min_df else {}),
        "metrics": {"per_fold": per_fold, "mean_std": summary, "pooled_out_of_fold": pooled},
        "error_examples": examples,
        "errors_total": int(len(wrong)),
        "errors_by_kind": {
            "false_positive": int((wrong["label"] == 0).sum()),
            "false_negative": int((wrong["label"] == 1).sum()),
        },
        "environment": environment(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    frame[["text_id", "fold", "label", "prob_ai"]].to_csv(out_dir / "oof_predictions.csv", index=False)
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    m = report["metrics"]["mean_std"]
    d = report["data"]
    lines = [
        f"# Baseline evaluation: {report['model']['name']}",
        "",
        f"Dataset: {report['dataset']}. Created {report['created_at']}.",
        "",
    ]
    if not report["reproduces_competition"]:
        lines += ["These numbers come from a synthetic fixture. They say nothing about the competition.", ""]
    lines += [
        "| Metric | Mean | Std (5 folds) |",
        "|---|---|---|",
        *[f"| {k} | {m[k]['mean']:.4f} | {m[k]['std']:.4f} |" for k in METRICS],
        "",
        f"Rows: {d['rows_raw']} raw, {d['rows_unique']} unique after merging duplicates "
        f"({d['duplicate_rows_removed']} removed, {d['conflicting_label_texts_dropped']} conflicting texts dropped).",
    ]
    if d.get("test_present"):
        lines.append(
            f"Test rows also in train: {d['test_rows_also_in_train']} of {d['test_rows']} "
            f"({d['test_unique_texts_also_in_train']} unique texts)."
        )
    k = report["errors_by_kind"]
    lines += [
        "",
        f"Misclassified out-of-fold at 0.5: {report['errors_total']} "
        f"({k['false_positive']} human texts scored as AI, {k['false_negative']} AI texts scored as human). "
        "Most confident errors:",
        "",
    ]
    for e in report["error_examples"]:
        lines.append(f"- {e['kind']}, p(AI)={e['prob_ai']}, fold {e['fold']}: \"{e['snippet']}\"")
    return "\n".join(lines) + "\n"
