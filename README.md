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
├── Competition 2/              # [In progress]
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

## Technical Stack

- Python, pandas, numpy
- scikit-learn, XGBoost, LightGBM, CatBoost
- TensorFlow/Keras
- Optuna
- matplotlib, seaborn

## Running the Code

```bash
cd "Competition 1"
pip install -r requirements.txt
jupyter notebook
```

Start with `EDA.ipynb`, then explore model notebooks in any order.

## Notes

- Model weights and generated artifacts are excluded from version control.
- Notebooks share common preprocessing through `helpers/process_data.py` and `helpers/nlp.py`.
- Random seeds and train/test splits are fixed for reproducible comparisons across experiments.
