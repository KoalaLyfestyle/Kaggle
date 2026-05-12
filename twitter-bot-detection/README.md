# Twitter Bot Detection

Binary classification competition to identify bot accounts on Twitter using account metadata and text features.

## Problem

Predict whether a Twitter account is a bot (`target = 1`) or a genuine user (`target = 0`). The dataset contains account-level features including engagement metrics, profile settings, and text fields such as description and screen name.

## Dataset

| Split | Rows |
|-------|------|
| Train | 26,206 |
| Test  | 11,232 |

**Features:**
- **Numerical:** `followers_count`, `friends_count`, `favourites_count`, `statuses_count`, `average_tweets_per_day`, `account_age_days`
- **Binary:** `default_profile`, `default_profile_image`, `geo_enabled`, `verified`
- **Categorical:** `lang` (one-hot encoded)
- **Text:** `description`, `screen_name`, `location`
- **Temporal:** `created_at` (extracted to `account_age_days`)

## Approach

### 1. Exploratory Data Analysis (`EDA.ipynb`)

Examined feature distributions, class balance, and correlations. Key findings:
- `statuses_count` is strongly correlated with `average_tweets_per_day` × `account_age_days` — one is redundant
- `id` correlates negatively with `account_age_days` (−0.74), confirming Twitter's sequential ID assignment
- Most numeric features are right-skewed; log transformation improves distribution for linear models

### 2. Feature Engineering

The `helpers/process_data.py` pipeline handles:
- Median imputation for numerical features, mode imputation for binary features
- One-hot encoding for `lang` with rare-language grouping
- Engineered ratio features: `followers_per_day`, `friends_per_day`, `favourites_per_tweet`
- Boolean flags from text fields: `description_has_url`, `screen_name_has_numbers`

The `helpers/nlp.py` pipeline adds text features via:
- Word-level TF-IDF (unigrams + bigrams) on combined `description`, `screen_name`, `location`
- Character-level TF-IDF (3–5 char n-grams)
- TruncatedSVD dimensionality reduction to 300 components
- GPU acceleration via cuML when available, with sklearn fallback

### 3. Models

Each base model was trained with 5-fold stratified cross-validation. Hyperparameters were tuned with [Optuna](https://optuna.org/).

| Notebook | Description |
|----------|-------------|
| `XGBoost.ipynb` | XGBoost baseline + Optuna tuning |
| `XGBoost+NLP.ipynb` | XGBoost with structured + NLP features |
| `LightGBM.ipynb` | LightGBM baseline + Optuna tuning |
| `LightGBM+NLP.ipynb` | LightGBM with NLP features |
| `CatBoost.ipynb` | CatBoost baseline + Optuna tuning |
| `CatBoost+NLP.ipynb` | CatBoost with NLP features |
| `NN_STRUCT_MLP.ipynb` | Neural network on structured features only |
| `NN_Text_MLP.ipynb` | Neural network on text features only |
| `NN_ALL_MLP.ipynb` | Neural network combining all features |

### 4. Ensembling

**Weighted Ensemble** (`weighted_ensemble.ipynb`): Combines Optuna-tuned XGBoost, LightGBM, and CatBoost predictions. Ensemble weights are optimised with Optuna over 100 trials.

**Meta-Ensemble + Pseudo-Labeling** (`meta_ensemble.ipynb`): Stacks base model out-of-fold predictions as features for a Logistic Regression meta-learner. Optionally extends training with high-confidence test pseudo-labels.

### 5. Evaluation

**Cross-Model Analysis** (`Eval.ipynb`): Loads all trained models and Optuna studies to compare performance, analyze feature importances, and visualize optimization history.

### Results

| Model | CV AUC | CV Accuracy |
|-------|--------|-------------|
| XGBoost (Optuna) | ~0.934 | ~0.93 |
| Weighted Ensemble (Optuna) | ~0.935 | ~0.93 |

## Repository Structure

```
twitter-bot-detection/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── helpers/
│   ├── process_data.py        # Structured feature preprocessing pipeline
│   ├── nlp.py                 # TF-IDF + SVD text feature extraction (GPU/CPU)
│   ├── nlp_cache.py           # Cached NLP feature loader
│   └── nlp_cpu.py             # CPU-only NLP fallback
├── models/
│   ├── XGBoost.ipynb
│   ├── XGBoost+NLP.ipynb
│   ├── LightGBM.ipynb
│   ├── LightGBM+NLP.ipynb
│   ├── CatBoost.ipynb
│   ├── CatBoost+NLP.ipynb
│   ├── NN_STRUCT_MLP.ipynb
│   ├── NN_Text_MLP.ipynb
│   ├── NN_ALL_MLP.ipynb
│   ├── weighted_ensemble.ipynb
│   └── meta_ensemble.ipynb
├── submissions/               # Generated CSV files for Kaggle submission
├── EDA.ipynb
├── Eval.ipynb
└── requirements.txt
```

## Running the Notebooks

Run from the `twitter-bot-detection/` directory so relative data paths resolve correctly.

1. Start with `EDA.ipynb` to understand the data
2. Run any individual model notebook — each is self-contained and uses `helpers/` for preprocessing
3. Run `weighted_ensemble.ipynb` or `meta_ensemble.ipynb` after training base models
4. Run `Eval.ipynb` to compare all trained models

GPU acceleration is detected automatically. To force CPU execution, set `tree_method='hist'` (XGBoost) and `device='cpu'` (LightGBM/CatBoost) in the respective notebooks.

## Results

**Competition**: BT4012 In-class Kaggle Competition 1 [2025]
- **Participants**: 117
- **Metric**: ROC-AUC
- **Evaluation**: Final ranking based on private leaderboard (70% of test data)
- **My result**: 75th place (private), dropped 53 places from public (was 22nd public)
- **Selected submission**: 0.94327
- **Best model (not selected)**: XGBoost+NLP Optuna at 0.94466 — would have placed ~43rd
- **Gap left on table**: 0.00139

The drop from 22nd to 75th was a clear case of overfitting the public leaderboard. I selected my XGBoost baseline over the XGBoost+NLP variant because the public scores were close and I didn't trust the NLP model to generalize. The final ranking was based entirely on the private set (70% of data), where the NLP model was significantly better. This taught me that when margins are tight and the public set is small, model selection must prioritize cross-validation and generalization over public leaderboard position.

## Reflection

This was my first structured Kaggle competition. The main lesson was that fraud detection models benefit both from domain understanding (feature engineering gave strong improvements) and from advanced techniques like pseudo-labelling and ensembling. Pseudo-labelling consistently improved CV ROC-AUC by approximately 1.7%, and the meta-ensemble outperformed all individual models.

The biggest failure was model selection. I left 0.00139 ROC-AUC on the table — enough to jump from 75th to ~43rd place — by not trusting my best CV model. The public leaderboard was only 30% of the data, and I overfit to it massively (dropping 53 places). I also had poor planning: I combined all experiment notebooks into one close to the submission date, which was a rush to rerun everything.

A full academic report is included as `report.pdf`.

## Dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optional for GPU NLP: `cupy cuml`
