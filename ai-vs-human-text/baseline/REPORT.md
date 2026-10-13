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
| TF-IDF + LogReg (this folder) | 4,212 unique texts | fixed folds over unique texts | 0.9926 ± 0.0015 | 0.245 | not submitted | rerun |
| TF-IDF + LogReg (notebook 1) | 4,712 rows, with duplicates | StratifiedKFold, seed 42 | 0.9940 ± 0.0013 | 0.231 | 0.98954 | historical |
| TF-IDF + LogReg (notebook 1) | 4,212 rows, deduplicated | StratifiedKFold, seed 42 | 0.9923 ± 0.0017 | 0.245 | none | historical |
| DeBERTa-v3-base | 4,712 rows, with duplicates | StratifiedKFold, seed 42 | 0.9994 ± 0.0002 | 0.050 | 0.99964 | historical, not reproduced |
| DeBERTa-v3-base | deduplicated | StratifiedKFold, seed 42 | 0.9988 ± 0.0013 | 0.071 | 0.99952 | historical, not reproduced |

The rerun baseline matches the historical deduplicated notebook run within
noise (0.9926 against 0.9923). DeBERTa ranks better (ROC-AUC about 0.999)
and its probabilities are much better calibrated (log loss 0.05 to 0.07
against 0.24).

## Checks

**Train-only fitting.** The baseline refits TF-IDF inside each fold, and a
test checks that no validation text reaches a fit. Notebook 1 also used a
Pipeline inside each fold, so its CV is clean on this point. The DeBERTa
notebook tokenizes with a pretrained tokenizer, which has no fitted state.

**Duplicates across splits.** The training data has 500 duplicate rows (306
texts repeated up to 5 times, all human). Most notebook experiments,
including the DeBERTa run that scored best, used the data with duplicates
and an ungrouped StratifiedKFold. Copies of the same text then land in both
training and validation folds, which inflates CV. The effect is visible:
removing duplicates lowers the TF-IDF CV from 0.9940 to 0.9923 and DeBERTa
from 0.9994 to 0.9988. The baseline merges duplicates before splitting, so
it does not have this leak.

**Train and test overlap.** 212 test rows (166 unique texts, 18% of the test
set) appear verbatim in the training data, all labelled human there. Any
model that memorises training texts gets these right for free, so leaderboard
scores overstate performance on new text.

**Leaderboard feedback.** The public leaderboard influenced model choice.
`../Evaluation.ipynb` ranks models by public score, `../top_10_models.csv` is
sorted by it, and the README has a "Top 5 by Kaggle Score" section. The final
choice between the two DeBERTa runs was made on CV, as the README says, but
the shortlist was shaped by public scores on 30% of about 1,300 test rows.
Treat the public scores as optimistic.

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
240 errors, 201 of them AI texts scored as human.

- AI scored as human (p = 0.17): "A strong linear relationship was observed
  between sheet count and speed loss, as reflected by an R-squared value of
  0.99 ..."
- AI scored as human (p = 0.17): "External service features are identified
  by querying various third-party services and search engines ..."
- Human scored as AI (p = 0.72): "This expansion will make our model more
  scalable and comprehensive, providing traders with valuable signals ..."
- Human scored as AI (p = 0.71): "In terms of future work, there are several
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
- The data has no recorded source or licence in this repository.
- No ensemble or search was run, by design.

## Next

- Rerun DeBERTa on `splits/folds.csv` (GPU needed) to get a like-for-like
  comparison.
- Calibrate the baseline's probabilities if they are used for anything other
  than ranking.
