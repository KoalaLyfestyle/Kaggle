# TF-IDF baseline for the AI-vs-human text track

A small, reproducible CPU baseline: word 1-2 gram TF-IDF and logistic
regression, cross-validated on fixed folds. It runs in seconds on a laptop and
needs no GPU, no downloads, and no API keys.

The notebooks in the parent folder stay as exploratory records. This folder is
the one path that is meant to be rerun.

## Run

```bash
cd ai-vs-human-text/baseline
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

python -m textbaseline evaluate      # 5-fold CV on splits/folds.csv -> runs/evaluate/report.json, report.md
python -m textbaseline train         # fit on all training texts -> runs/train/model.joblib, submission.csv
python -m pytest                     # smoke tests on the synthetic fixture
```

Data comes from `../data` by default. Use `--data-dir DIR` or set
`AIVH_DATA_DIR` to point elsewhere. The folder needs `train.csv` (`text`,
`label`: 0 human, 1 AI) and, for a submission, `test.csv` (`id`, `text`).

## What is fixed

- **Duplicates.** Texts are merged when they match after lowercasing and
  collapsing whitespace. Each text gets a `text_id` (SHA-1 of that form).
  Texts with conflicting labels are dropped. The competition data has 500
  duplicate rows, all human.
- **Folds.** `splits/folds.csv` assigns every unique text to one of 5 folds
  (stratified, seed 42). `evaluate` refuses to run if the manifest does not
  match the data. Regenerate it with `make-splits --force` only if the data
  changed on purpose.
- **Fitting.** The vectorizer and classifier are refitted inside each fold on
  the training folds only. A test checks this.
- **Versions.** `requirements.lock` pins every package for Python 3.12.

## Files

- `textbaseline/` - data loading, folds, model, evaluation, training, CLI
- `splits/folds.csv` - the fixed folds for the committed competition data
- `reports/tfidf-logreg/` - the report from the last run on the competition data
- `tests/` - smoke tests on a synthetic fixture (`tests/fixtures/make_fixture.py`)
- `REPORT.md` - baseline versus the DeBERTa experiment
- `MODEL_CARD.md` - what the baseline is for and how it fails
