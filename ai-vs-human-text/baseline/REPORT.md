# Baseline versus the DeBERTa experiment

This compares the reproducible TF-IDF baseline in this folder with the best
transformer experiment in the parent folder, `12_DeBERTa.ipynb`. Baseline
numbers were rerun on 13 October 2026. DeBERTa numbers are historical: they
come from `../comprehensive_model_comparison.csv` and `../kaggle_scores.csv`
and were **not reproduced** here (the notebook needs a GPU and a Colab drive
layout).

## Results

| Model | Data | Split | CV ROC-AUC | CV log loss | Public LB | Status |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg (this folder) | 4,162 unique texts | fixed folds, near-duplicates grouped | 0.9915 ± 0.0018 | 0.248 | not submitted | rerun |
| TF-IDF + LogReg (notebook 1) | 4,712 rows, with duplicates | StratifiedKFold, seed 42 | 0.9940 ± 0.0013 | 0.231 | 0.98954 | historical |
| TF-IDF + LogReg (notebook 1) | 4,212 rows, deduplicated | StratifiedKFold, seed 42 | 0.9923 ± 0.0017 | 0.245 | none | historical |
| DeBERTa-v3-base | 4,712 rows, with duplicates | StratifiedKFold, seed 42 | 0.9994 ± 0.0002 | 0.050 | 0.99964 | historical, not reproduced |
| DeBERTa-v3-base | deduplicated | StratifiedKFold, seed 42 | 0.9988 ± 0.0013 | 0.071 | 0.99952 | historical, not reproduced |

The rerun baseline matches the historical deduplicated notebook run within
noise (0.9915 against 0.9923), even though it also merges near-copies that
the notebook kept. DeBERTa ranks better (ROC-AUC about 0.999)
and its probabilities are much better calibrated (log loss 0.05 to 0.07
against 0.25).

## Checks

**Train-only fitting.** The baseline refits TF-IDF inside each fold, and a
test checks that no validation text reaches a fit. Notebook 1 also used a
Pipeline inside each fold, so its CV is clean on this point. The DeBERTa
notebook's tokenizer has no fitted state, but its training loop picks the
best epoch and stops early on the validation fold's ROC-AUC, then reports
that same fold's score (`12_DeBERTa.ipynb`, cell 15). Its CV numbers are
therefore somewhat optimistic.

**Duplicates across splits.** The training data has 500 exact duplicate rows
(306 texts repeated up to 5 times, all human), and 50 more once case,
spacing, and punctuation are ignored. Another 134 groups of texts share
their opening 100 characters, for example the same paragraph with a
citation added. Most notebook experiments,
including the DeBERTa run that scored best, used the data with duplicates
and an ungrouped StratifiedKFold. Copies of the same text then land in both
training and validation folds, which inflates CV. The effect is visible:
removing duplicates lowers the TF-IDF CV from 0.9940 to 0.9923 and DeBERTa
from 0.9994 to 0.9988. The baseline merges duplicates before splitting and
keeps each near-duplicate group in one fold. Paraphrases that change the
opening can still cross folds; nothing here detects them.

**Train and test overlap.** 212 test rows (166 unique texts, 18% of the 1,178
test rows) appear verbatim in the training data, all labelled human there;
230 rows once case, spacing, and punctuation are ignored. The test set also
repeats itself: 63 of its rows duplicate another test row. Any
model that memorises training texts gets these right for free, so leaderboard
scores overstate performance on new text.

**Leaderboard feedback.** The public leaderboard influenced model choice.
`../Evaluation.ipynb` ranks models by public score, `../top_10_models.csv` is
sorted by it, and the README has a "Top 5 by Kaggle Score" section. The final
pick was the deduplicated DeBERTa run, although both its CV (0.99876 against
0.99940) and its public score (0.99952 against 0.99964) were lower than the
run with duplicates. So it was chosen for the cleaner data, not for a higher
score. Its private score was 0.99943, against 0.99944 for the other run
(from the README). The shortlist was still shaped by public scores on 30% of
the 1,178 test rows. Treat the public scores as optimistic.

**Split provenance.** Notebook folds were drawn per notebook with
`StratifiedKFold(5, shuffle=True, random_state=42)` on whichever data version
that notebook loaded, so fold membership differs between the original and
deduplicated runs. The baseline stores its folds in `splits/folds.csv`,
keyed by text, so any later experiment can reuse exactly the same folds.

**Tuning.** The notebooks' Optuna studies have no sampler seed, so the tuned
variants (for example the public 0.99611 TF-IDF run) cannot be reproduced
exactly. The baseline uses the untuned settings on purpose.

## Error examples

From `reports/tfidf-logreg/report.json`, out-of-fold at a 0.5 threshold:
240 errors, 197 of them AI texts scored as human (one example per near-duplicate group).

- AI scored as human (p = 0.16): "External service features are identified
  by querying various third-party services and search engines ..."
- AI scored as human (p = 0.17): "A strong linear relationship was observed
  between sheet count and speed loss, as reflected by an R-squared value of
  0.99 ..."
- Human scored as AI (p = 0.83): "This expansion will make our model more
  scalable and comprehensive, providing traders with valuable signals ..."
- Human scored as AI (p = 0.73): "In terms of future work, there are several
  strategies to enhance our predictive models. Firstly, ..."

The baseline reacts to generic connective phrasing ("comprehensive",
"enhance", "Firstly"). Specific, technical AI text slips through, and formal
human text gets flagged. See `MODEL_CARD.md`.

## Limitations

- DeBERTa numbers were not rerun. They come from saved CSVs, and the
  notebook's saved outputs belong to another experiment, so they cannot be
  checked against the notebook itself.
- Even the deduplicated historical runs used notebook-specific folds, so the
  comparison with the baseline is close but not exact.
- The near-duplicate rule (same first 100 normalised characters) is a
  heuristic. It misses paraphrases and may group unrelated texts that share
  a boilerplate opening; grouping too much only makes CV stricter.
- The data has no recorded source or licence in this repository.
- No ensemble or search was run, by design.

## Next

- Rerun DeBERTa on `splits/folds.csv` (GPU needed) to get a like-for-like
  comparison.
- Calibrate the baseline's probabilities if they are used for anything other
  than ranking.
