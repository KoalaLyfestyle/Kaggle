# Kaggle Competition Solutions

End-to-end machine learning pipelines and experiments from Kaggle competitions.

## Repository Structure

```
Kaggle/
├── Competition 1/              # Twitter Bot Detection (Binary Classification)
│   ├── data/                   # Competition datasets
│   ├── helpers/                # Reusable preprocessing and NLP modules
│   ├── models/                 # Experiment notebooks (one per model/approach)
│   ├── submissions/            # Kaggle submission files
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Eval.ipynb             # Cross-model evaluation and comparison
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Competition write-up
├── Competition 2/              # AI vs Human Text Classification (Binary Classification)
│   ├── data/                   # Competition datasets
│   ├── models/                 # Experiment notebooks (one per model/approach)
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Eval.ipynb             # Cross-model evaluation and comparison
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Competition write-up
└── README.md                  # This file
```

## Competition 1: Twitter Bot Detection

**Task:** Binary classification to predict whether a Twitter account is a bot.

**Dataset:** 26,206 training samples, 11,232 test samples. Features include numeric engagement metrics, binary profile settings, categorical language codes, and text fields (description, screen name, location).

### Experiment Notebooks

Each notebook is self-contained and uses the same data loading and preprocessing pipeline from `helpers/`.

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Data profiling, distributions, correlations, class balance |
| `models/CatBoost.ipynb` | Gradient boosting with native categorical handling |
| `models/CatBoost+NLP.ipynb` | CatBoost with TF-IDF text features |
| `models/LightGBM.ipynb` | LightGBM baseline |
| `models/LightGBM+NLP.ipynb` | LightGBM with text features |
| `models/XGBoost.ipynb` | XGBoost with Optuna hyperparameter tuning |
| `models/XGBoost+NLP.ipynb` | XGBoost with NLP pipeline |
| `models/NN_STRUCT_MLP.ipynb` | Neural network on structured features |
| `models/NN_Text_MLP.ipynb` | Neural network on text features only |
| `models/NN_ALL_MLP.ipynb` | Neural network combining all features |
| `models/weighted_ensemble.ipynb` | Weighted averaging of base models |
| `models/meta_ensemble.ipynb` | Stacking ensemble with pseudo-labeling |
| `Eval.ipynb` | Model comparison, Optuna analysis, feature importance |

### Methods

- **Feature engineering:** numerical interactions, ratio features, datetime extraction, TF-IDF with dimensionality reduction
- **Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP neural networks
- **Validation:** 5-fold stratified cross-validation with consistent splits across all experiments
- **Optimization:** Optuna for hyperparameter search
- **Advanced:** pseudo-labeling (semi-supervised), stacking ensembles

## Competition 2: AI vs Human Text Classification

**Task:** Binary classification to distinguish AI-generated text from human-written text.

**Dataset:** 4,212 training samples (after deduplication), 1,178 test samples. Raw text with engineered structural features.

### Experiment Notebooks

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Text-specific EDA: length distributions, readability scores, entropy |
| `Processing.ipynb` | Leakage-free feature engineering pipeline |
| `1-Log-TFIDF.ipynb` | Logistic regression with TF-IDF baseline |
| `2-Ridge-TFIDF.ipynb` | Ridge regression with TF-IDF |
| `3-XGB-STRUCT.ipynb` | XGBoost on structural text features |
| `4-XGB-ALL.ipynb` | XGBoost on all features |
| `5-MLP-ALL.ipynb` | Multilayer perceptron on all features |
| `6-BiLSTM.ipynb` | Bidirectional LSTM on tokenized text |
| `7-CharCNN.ipynb` | Character-level CNN |
| `8-HybridNN1.ipynb` | Hybrid architecture experiments |
| `9-HybridNN2.ipynb` | Second hybrid architecture variant |
| `10-Ensemble.ipynb` | Ensemble methods |
| `11-RoBERTa.ipynb` | Fine-tuned RoBERTa transformer |
| `12-DeBERTa.ipynb` | Fine-tuned DeBERTa-v3 transformer (best single model) |
| `Evaluation.ipynb` | Comprehensive cross-model comparison and analysis |

### Methods

- **Feature engineering:** text statistics (length, entropy, readability), TF-IDF, character n-grams
- **Models:** Logistic Regression, Ridge, XGBoost, MLP, BiLSTM, CharCNN, RoBERTa, DeBERTa
- **Validation:** 5-fold stratified cross-validation
- **Optimization:** Optuna for traditional ML hyperparameters
- **Advanced:** ensemble averaging, weighted voting, progressive complexity pipeline

## Technical Stack

- Python, pandas, numpy
- scikit-learn, XGBoost, LightGBM, CatBoost
- TensorFlow/Keras
- Hugging Face Transformers
- Optuna
- matplotlib, seaborn

## Running the Code

Each competition is self-contained. Navigate to the competition directory and install dependencies:

```bash
cd "Competition 1"   # or "Competition 2"
pip install -r requirements.txt
jupyter notebook
```

Start with `EDA.ipynb`, then explore model notebooks in any order.

## Notes

- Model weights and generated artifacts are excluded from version control.
- Notebooks in Competition 1 share preprocessing through `helpers/`.
- Notebooks in Competition 2 use `Processing.ipynb` as the reference pipeline.
- Random seeds and train/test splits are fixed for reproducible comparisons across experiments.
