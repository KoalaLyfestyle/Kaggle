# The competition notebooks

The notebooks are the records of two BT4012 in-class Kaggle competitions
(2025). They are kept as exploratory records. Most of them were run on Colab
or on one laptop, and several still point at a Google Drive folder or a local
path, so they do not rerun from a fresh clone as they are. The one path that
is meant to be rerun is [`ai-vs-human-text/baseline/`](../ai-vs-human-text/baseline/README.md).

## twitter-bot-detection

Binary classification of Twitter accounts as bots or humans, from account
metadata and text fields.

- Exploratory analysis: `EDA.ipynb`.
- Shared preprocessing and text features: `helpers/` (`process_data.py`,
  `nlp.py`, `nlp_cpu.py`, `nlp_cache.py`).
- Models in `models/`: XGBoost, LightGBM, and CatBoost, each with and without
  text features; three MLPs (structured, text, all features); a weighted
  ensemble and a meta-ensemble (`meta_ensemble.ipynb`, which also tries
  pseudo-labelling).
- Comparison: `Eval.ipynb`.

Full write-up: [twitter-bot-detection/README.md](../twitter-bot-detection/README.md).

## ai-vs-human-text

Binary classification of short texts as AI-written or human-written.

- Exploratory analysis and preprocessing: `EDA.ipynb`, `Processing.ipynb`.
- Twelve numbered model notebooks: TF-IDF with logistic regression and
  ridge (1, 2), XGBoost on structured and all features (3, 4), an MLP (5),
  BiLSTM (6), CharCNN (7), two hybrid networks (8, 9), ensembles (10),
  RoBERTa (11), and DeBERTa-v3 (12).
- Comparison: `Evaluation.ipynb` and the CSV files in the project folder,
  for example `comprehensive_model_comparison.csv` and `fold_level_metrics.csv`.
  [`EVALUATION_README.md`](../ai-vs-human-text/EVALUATION_README.md) explains them.

Full write-up: [ai-vs-human-text/README.md](../ai-vs-human-text/README.md).

## Running a notebook

```bash
cd twitter-bot-detection   # or ai-vs-human-text
pip install -r requirements.txt
jupyter notebook
```

Start with `EDA.ipynb`. Expect to edit data paths in some notebooks, and
expect the transformer notebooks to need a GPU.

## Notes

- Model weights are not committed. The comparison CSVs are, and so are the
  bot detection submission files. The AI-text project has no committed
  submissions.
- Each project folder has the academic report submitted for grading
  (`report.pdf`).
- Notebook folds were drawn per notebook with `StratifiedKFold(5,
  shuffle=True, random_state=42)` on whichever data version it loaded. The
  AI-text Optuna studies in notebooks 1 to 5 have no sampler seed, so those
  tuned results cannot be reproduced exactly. The bot detection studies use
  `TPESampler(seed=42)`. See
  [the baseline report](../ai-vs-human-text/baseline/REPORT.md#checks).
