# Evaluation Notebook - Updated for Dual Directory Structure

## Overview

The `Evaluation.ipynb` notebook has been updated to handle models from **two sources**:

1. **`experiments/`** - Models trained on **deduplicated data**
2. **`experiments_old/`** - Models trained on **original (non-deduplicated) data**

This allows you to compare model performance across different data preprocessing approaches and determine the impact of deduplication.

---

## Key Changes

### 1. **Dual Directory Support**

- Automatically loads metrics from both `experiments/` and `experiments_old/`
- Tracks which directory each model came from
- Labels models with `[deduplicated]` or `[original]` for easy identification

### 2. **Kaggle Scores Integration**

The notebook supports multiple ways to load Kaggle submission scores:

#### **Preferred Method: Root `kaggle_scores.csv`**

Create a file `kaggle_scores.csv` in the root directory with this format:

```csv
model_name,source,score
exp_1_tfidf_log,experiments,0.99123
exp_1_tfidf_log,experiments_old,0.98987
exp_2_tfidf_ridge,experiments,0.99234
exp_2_tfidf_ridge_optuna,experiments,0.99345
```

**Columns:**

- `model_name`: Experiment name (e.g., `exp_1_tfidf_log`, `exp_12_deberta`)
- `source`: Either `experiments` or `experiments_old`
- `score`: Kaggle public leaderboard ROC-AUC score

A template file `kaggle_scores_template.csv` has been created for you. Simply:

1. Copy it to `kaggle_scores.csv`
2. Fill in the score column with your actual Kaggle submission scores

#### **Fallback Methods:**

If `kaggle_scores.csv` doesn't exist, the notebook will look for:

- `experiments/kaggle_results.csv` (for deduplicated models)
- `experiments_old/kaggle_results.csv` (for original data models)

### 3. **No Model Loading**

- Only analyzes `metrics.json` files and submission CSVs
- Does NOT load any `.pkl` or model weight files
- Fast execution, no memory issues

### 4. **Enhanced Visualizations**

All plots now distinguish between deduplicated and original data models:

- **Blue** bars/points = Deduplicated data
- **Orange** bars/points = Original data
- Side-by-side comparisons to see impact of deduplication

### 5. **Data Version Comparison**

New analysis sections compare:

- Average performance by data version
- Generalization gaps between versions
- Best models from each data source

---

## How to Use

### Step 1: Organize Your Experiments

Ensure your directory structure looks like this:

```
ai-vs-human-text/
├── experiments/              # Deduplicated data models
│   ├── exp_1_tfidf_log/
│   │   ├── metrics.json
│   │   └── exp_1_tfidf_log_submission.csv
│   ├── exp_4_all_xgboost/
│   ├── exp_12_deberta/
│   └── ...
│
├── experiments_old/          # Original data models
│   ├── exp_1_tfidf_log/
│   │   ├── metrics.json
│   │   └── exp_1_tfidf_log_submission.csv
│   ├── exp_10_stacked_ensemble/
│   ├── exp_11_roberta/
│   └── ...
│
├── kaggle_scores.csv         # Your Kaggle submission scores
└── Evaluation.ipynb
```

### Step 2: Prepare Kaggle Scores

**Option A: Use the template (Recommended)**

```bash
cp kaggle_scores_template.csv kaggle_scores.csv
# Edit kaggle_scores.csv and fill in the scores
```

**Option B: Create from scratch**

```csv
model_name,source,score
exp_12_deberta,experiments,0.99456
exp_11_roberta,experiments_old,0.99234
exp_10_stacked_ensemble,experiments_old,0.99123
...
```

### Step 3: Run the Notebook

Simply execute all cells in `Evaluation.ipynb`. The notebook will:

1. ✅ Load all metrics from both directories
2. ✅ Match with Kaggle scores
3. ✅ Generate comprehensive visualizations
4. ✅ Export analysis CSVs
5. ✅ Provide recommendations

---

## Output Files

After running the notebook, you'll find:

1. **`comprehensive_model_comparison.csv`**

   - All models with CV and Kaggle metrics
   - Includes source (experiments vs experiments_old)

2. **`data_version_comparison.csv`**

   - Statistical comparison between deduplicated and original data
   - Shows impact of deduplication

3. **`final_model_rankings.csv`**

   - Top 10 models ranked by composite score
   - Weighted by Kaggle performance, CV, stability, and generalization

4. **`model_family_summary.csv`**

   - Performance by model architecture (Ensemble, Hybrid, BERT, etc.)

5. **`top_10_models.csv`**

   - Quick reference for best performers

6. **`fold_level_metrics.csv`**
   - Detailed fold-by-fold performance for variance analysis

---

## Interpretation Guide

### CV vs Kaggle Scores

- **High correlation**: CV is a reliable predictor of test performance
- **Low correlation**: Public LB may not represent private LB well

### Generalization Gap

- **Negative gap** (Kaggle > CV): Model generalizes well to test data
- **Positive gap** (CV > Kaggle): Potential overfitting to training folds
- **Goal**: Minimize absolute gap for consistent performance

### Data Version Impact

Compare deduplicated vs original:

- Which performs better on Kaggle?
- Which has lower variance across folds?
- Are certain model families more sensitive to deduplication?

---

## Troubleshooting

### "No Kaggle scores found!"

- Ensure `kaggle_scores.csv` exists in root directory
- Or create `experiments/kaggle_results.csv` and `experiments_old/kaggle_results.csv`
- Check column names are correct: `model_name`, `source`, `score`

### "No metrics file found for..."

- Ensure each experiment directory has `metrics.json` or `metrics_optuna.json`
- Check file paths are correct
- Follow the MODEL_TRACKING.md guide for proper experiment structure

### Some models missing from visualizations

- Check if Kaggle scores exist for those models
- Models without Kaggle scores will only appear in CV analysis sections

---

## Next Steps

1. **Fill in Kaggle Scores**: Update `kaggle_scores.csv` with actual submission results
2. **Run Analysis**: Execute all cells in `Evaluation.ipynb`
3. **Review Outputs**: Check the generated CSV files for insights
4. **Compare Data Versions**: Determine if deduplication helped
5. **Select Final Model**: Based on composite rankings and data version analysis
6. **Ensemble Strategy**: Consider combining top models from both data sources

---

## Questions?

Refer to:

- `experiments/MODEL_TRACKING.md` - For experiment organization standards
- `experiments_old/MODEL_TRACKING.md` - For legacy experiment details
- This notebook's markdown cells - For detailed explanations of each analysis section

---

**Last Updated**: November 8, 2025
**Notebook Version**: 2.0 (Dual Directory Support)
