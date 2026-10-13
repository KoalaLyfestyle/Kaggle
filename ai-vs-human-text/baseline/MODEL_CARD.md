# Model card: tfidf-word12-logreg

## What it is

Word unigram and bigram TF-IDF (10,000 features, `min_df=2`, `max_df=0.95`,
sublinear term frequency) feeding a logistic regression (`liblinear`,
`C=1`). It outputs the probability that a text was written by an AI.

It is a reference point, not the best model. It exists so that later
experiments have a fast, reproducible number to beat.

## Data

The BT4012 in-class Kaggle competition training set in `../data/train.csv`:
4,712 rows, 4,212 unique texts after merging duplicates, 31% AI. The texts
are short paragraphs; the examples seen include project-report passages and
product reviews. Where the data came from and its licence are not
recorded in this repository.

## Performance

5-fold cross-validation on unique texts (`reports/tfidf-logreg/report.json`):

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.9926 | 0.0015 |
| Log loss | 0.2445 | 0.0060 |
| Accuracy at 0.5 | 0.943 | 0.013 |
| F1 at 0.5 | 0.914 | 0.022 |

No competition leaderboard score was produced for this exact run.

## Expected failure modes

- **AI text written in a plain, specific style is missed.** Most errors (201
  of 240) are AI texts scored as human. The most confident misses describe
  concrete steps, numbers, or tools ("we generated image embeddings using
  ResNet50 ..."). The model keys on generic phrasing, not on authorship.
- **Polished human writing is flagged.** The 39 false positives are human
  texts with formal connectives ("comprehensive", "enhance", "Firstly ...").
  Careful non-native or edited writing is at risk.
- **Probabilities are not calibrated.** Log loss is 0.24 against ROC-AUC
  0.99: the ranking is good, the probability values are not. Do not read 0.7
  as "70% likely". Use it for ranking, or calibrate on held-out data first.
- **Narrow domain.** Trained on one course's texts and one set of generators.
  Expect a large drop on other topics, other languages, newer models, or
  texts that were paraphrased or lightly edited by a human.
- **Short texts.** Below a sentence or two there is little signal.

## Not for

Grading, disciplinary decisions, or any judgement about a person. A score is
a weak signal about a text, not evidence about who wrote it.
