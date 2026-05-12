# Kaggle Competition Solutions

End-to-end machine learning pipelines from Kaggle competitions, organized as independent, reproducible experiments.

## twitter-bot-detection

Binary classification to detect Twitter bot accounts from account metadata and text fields. The pipeline covers exploratory data analysis, feature engineering (numerical interactions, TF-IDF text features), and a progression of models from logistic regression through gradient boosting (XGBoost, LightGBM, CatBoost) to neural networks and stacked ensembles with pseudo-labeling. All experiments use consistent 5-fold stratified cross-validation and shared preprocessing modules.

**Result**: 75th of 117 participants. The main lesson was in model selection — I overfit the public leaderboard (dropped 53 places to private) and left 0.00139 ROC-AUC on the table by not selecting my best CV model.

See [twitter-bot-detection/README.md](twitter-bot-detection/README.md) for the full write-up.

## ai-vs-human-text

Binary classification to distinguish AI-generated text from human-written text. The approach moves from TF-IDF baselines through structured feature engineering (readability scores, POS tags, entropy), gradient boosting, and neural networks (MLP, BiLSTM, CharCNN) to fine-tuned transformers (RoBERTa, DeBERTa-v3). Emphasis on pipeline-based training to prevent data leakage and systematic experiment tracking across 12 model variants.

**Result**: 32nd of 113 participants. Applied the lessons from Competition 1: the public-to-private drop shrank from 53 to 13 places, and the model selection gap fell from 0.00139 to 0.00001. The best single model (DeBERTa) outperformed all ensembles on the private set.

See [ai-vs-human-text/README.md](ai-vs-human-text/README.md) for the full write-up.

## Running the Code

Each competition is self-contained. Navigate to the project directory and install dependencies:

```bash
cd twitter-bot-detection   # or ai-vs-human-text
pip install -r requirements.txt
jupyter notebook
```

Start with `EDA.ipynb`, then explore model notebooks in any order.

## Notes

- Model weights and generated artifacts are excluded from version control.
- Random seeds and train/test splits are fixed for reproducible comparisons across experiments.
