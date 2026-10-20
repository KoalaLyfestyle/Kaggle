"""Draw the cross-validation chart on the README front page.

Run from the repository root:

    python docs/media/make_results_chart.py

Inputs (all committed):
- ai-vs-human-text/comprehensive_model_comparison.csv: historical notebook
  runs, mean and standard deviation of ROC-AUC over 5 folds.
- ai-vs-human-text/baseline/reports/tfidf-logreg/report.json: the
  reproducible baseline, rerun by `python -m textbaseline evaluate`.

Rows left out: the stacked ensemble with no CV score, the weighted-average
ensemble (its CV has no fold spread), and Optuna rows whose scores are
identical to the untuned row of the same model. Rows the CSV marks as
tuned are labelled "(tuned)". Error bars are cut at 1.0.

Needs pandas and matplotlib (the committed images were drawn with
matplotlib 3.11.2): `pip install matplotlib==3.11.2 pandas`.

Outputs: docs/media/cv-results-light.png and cv-results-dark.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "ai-vs-human-text" / "comprehensive_model_comparison.csv"
REPORT = ROOT / "ai-vs-human-text" / "baseline" / "reports" / "tfidf-logreg" / "report.json"
OUT = Path(__file__).resolve().parent

NAMES = {
    "exp_10_stacked_ensemble_legacy": "Stacked ensemble",
    "All Features + MLP": "All features + MLP",
    "All Features + XGBoost": "All features + XGBoost",
    "Structured Features + XGBoost": "Structured features + XGBoost",
    "TF-IDF + Logistic Regression": "TF-IDF + LogReg",
    "TF-IDF + Ridge Classifier": "TF-IDF + Ridge",
}

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "orig": "#0969da", "dedup": "#bf5700", "rerun": "#1a7f37"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "orig": "#4493f8", "dedup": "#f0883e", "rerun": "#3fb950"},
}


def load() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(CSV)
    d = d[d["cv_roc_auc"].notna() & (d["cv_roc_auc_std"] > 0)]
    d = d[~d["display_name"].str.contains("Optuna")]
    d = d.assign(name=d["display_name"].replace(NAMES))
    tuned = d["optimized"].astype(str).isin(["True", "T"])
    d.loc[tuned, "name"] = d.loc[tuned, "name"] + " (tuned)"
    rerun = json.loads(REPORT.read_text())["metrics"]["mean_std"]["roc_auc"]
    return d, rerun


def draw(d: pd.DataFrame, rerun: dict, theme: str) -> Path:
    c = THEMES[theme]
    order = d.groupby("name")["cv_roc_auc"].max().sort_values().index.tolist()
    rerun_label = "TF-IDF + LogReg, rerun"
    labels = order[: order.index("TF-IDF + LogReg")] + [rerun_label] + order[order.index("TF-IDF + LogReg"):]
    y = {name: i for i, name in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=110)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    for version, colour, marker, dy, label in [
        ("original", c["orig"], "o", 0.15, "Notebook run, data with duplicates"),
        ("deduplicated", c["dedup"], "s", -0.15, "Notebook run, deduplicated data"),
    ]:
        rows = d[d["data_version"] == version]
        # ROC-AUC cannot exceed 1, so the upper bar stops there.
        upper = (1.0 - rows["cv_roc_auc"]).clip(upper=rows["cv_roc_auc_std"])
        ax.errorbar(rows["cv_roc_auc"], [y[n] + dy for n in rows["name"]], xerr=[rows["cv_roc_auc_std"], upper],
                    fmt=marker, color=colour, ecolor=colour, elinewidth=1.6, capsize=3, markersize=6, label=label)
    ax.errorbar([rerun["mean"]], [y[rerun_label]], xerr=[rerun["std"]], fmt="D", color=c["rerun"],
                ecolor=c["rerun"], elinewidth=1.8, capsize=3, markersize=7,
                label="Reproducible baseline, rerun (near-duplicates grouped)")

    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("5-fold cross-validation ROC-AUC (mean ± 1 std)", color=c["fg"])
    ax.set_title("AI vs human text: cross-validation score per model", color=c["fg"], loc="left", fontsize=12)
    ax.tick_params(colors=c["muted"])
    for lab in ax.get_yticklabels():
        lab.set_color(c["fg"])
        if lab.get_text() == rerun_label:
            lab.set_fontweight("bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(c["fg"])
    fig.text(0.01, 0.01, "Notebook runs are historical and were not rerun. The rerun sits next to its notebook row. Folds differ between rows.",
             color=c["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = OUT / f"cv-results-{theme}.png"
    fig.savefig(path, facecolor=c["bg"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    data, rerun = load()
    for name in THEMES:
        print(draw(data, rerun, name))
