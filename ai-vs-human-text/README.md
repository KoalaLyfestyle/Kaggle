# AI vs Human Text Classification - Kaggle Competition

**Competition**: BT4012 In-Class Kaggle Competition (2025)  
**Task**: Binary classification to distinguish AI-generated text (label=1) from human-written text (label=0)  
**Primary Metric**: ROC-AUC  
**Final Performance**: 0.99964 ROC-AUC (DeBERTa-v3)

---

## Table of Contents

1. [Competition Overview](#competition-overview)
2. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
3. [Cross-Validation Strategy](#cross-validation-strategy)
4. [Model Development Pipeline](#model-development-pipeline)
5. [Experiments Summary](#experiments-summary)
6. [Key Machine Learning Techniques](#key-machine-learning-techniques)
7. [Best Performing Models](#best-performing-models)
8. [Ensemble Methods](#ensemble-methods)
9. [Lessons Learned](#lessons-learned)
10. [Reproducibility](#reproducibility)

---

## Competition Overview

### Dataset

- **Train**: 4,712 samples (before deduplication)
  - After deduplication: 4,212 samples
  - Class distribution: 64.89% Human (2,733), 35.11% AI (1,479)
- **Test**: 1,178 samples (public leaderboard)

### Approach Philosophy

This project followed a systematic, experiment-driven approach with:

- Comprehensive exploratory data analysis (EDA)
- Structured feature engineering pipeline
- Pipeline-based model development (prevents data leakage)
- Consistent cross-validation strategy across all models
- Rigorous tracking of metrics and artifacts
- Progressive complexity: baseline → traditional ML → deep learning → transformers → ensembles

---

## Data Preprocessing & Feature Engineering

### 1. Exploratory Data Analysis (`EDA.ipynb`)

Conducted comprehensive analysis to identify discriminative features:

**Key Findings:**

- AI texts showed slightly lower minimum word counts and character lengths
- Higher type-token ratio in AI texts (greater lexical variety)
- Lower Flesch Reading Ease in AI texts (more complex)
- Higher Shannon entropy in AI texts (less predictable word usage)
- Moderate class imbalance (31.39% AI vs 68.61% Human)

**Statistical Tests:**

- Performed independent t-tests on all features
- Identified features with significant differences (p < 0.05)
- Created correlation heatmaps to understand feature relationships

### 2. Feature Engineering Pipeline (`Processing.ipynb`)

Developed a comprehensive, leakage-free feature extraction pipeline:

#### Feature Groups

**Basic Text Features (12 features):**

- `char_len`: Total character count
- `word_count`: Total word count
- `unique_word_count`: Number of unique words
- `type_token_ratio`: Unique words / Total words (lexical diversity)
- `sentence_count`: Number of sentences (period-based)
- `avg_sentence_length`: Average words per sentence
- `punctuation_count` / `punctuation_ratio`: Punctuation usage
- `uppercase_count` / `uppercase_ratio`: Capitalization patterns
- `stopword_count` / `stopword_ratio`: Stopword usage

**Readability Features (3 features):**

- `flesch_reading_ease`: Flesch Reading Ease score
- `flesch_kincaid_grade`: Grade level readability
- `shannon_entropy`: Word distribution entropy (predictability measure)

**Part-of-Speech Features (18 features):**

- `pos_adj`, `pos_adp`, `pos_adv`, `pos_aux`, `pos_cconj`, `pos_det`, `pos_intj`
- `pos_noun`, `pos_num`, `pos_part`, `pos_pron`, `pos_propn`, `pos_punct`
- `pos_sconj`, `pos_sym`, `pos_verb`, `pos_x`, `pos_space`
- Computed using spaCy's `en_core_web_sm` model

**Total**: 33 engineered features, all computed independently per sample (no data leakage)

#### Data Leakage Prevention

✅ **Safe Practices:**

- All features computed from individual text samples only
- No cross-sample statistics (e.g., global mean, std)
- Pre-defined resources only (stopword lists, POS taggers)
- Formula-based calculations (readability scores)

⚠️ **Features Requiring Careful Handling:**

- TF-IDF vectorization: Fit separately per CV fold
- Feature scaling: Fit on training folds only
- Dimensionality reduction: Applied within pipelines

**Output Files:**

- `data/train_w_struct.csv`: Train data with all engineered features
- `data/test_w_struct.csv`: Test data with all engineered features

---

## Cross-Validation Strategy

### Stratified K-Fold Cross-Validation

**Consistent across ALL experiments:**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Configuration:**

- **Method**: Stratified K-Fold (preserves class distribution)
- **K**: 5 folds
- **Random State**: 42 (ensures reproducibility and fold alignment)
- **Shuffle**: True (randomizes sample order before splitting)

**Why This Matters:**

1. **Class Balance**: Each fold maintains ~35% AI, ~65% Human distribution
2. **Fold Alignment**: Same random_state=42 ensures all models use identical splits
3. **Reproducibility**: Results are deterministic and comparable
4. **Ensemble Readiness**: Out-of-fold predictions from different models align perfectly

**Evaluation Metrics (Per Fold):**

- Accuracy
- Precision
- Recall
- F1 Score
- **ROC-AUC** (primary metric)
- Log Loss

**Aggregation:**

- Mean ± Std across 5 folds for all metrics
- Best fold identification based on ROC-AUC
- Submission: Averaged predictions from all 5 fold models

---

## Model Development Pipeline

### Pipeline-Based Approach

**Standard Pattern Used Across All Traditional ML Models:**

```python
from sklearn.pipeline import Pipeline

def create_pipeline(vectorizer_params, model_params):
    return Pipeline([
        ('vectorizer', TfidfVectorizer(**vectorizer_params)),
        ('model', Model(**model_params))
    ])

# Per-fold training (prevents data leakage)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    pipeline = create_pipeline(vectorizer_params, model_params)
    pipeline.fit(X[train_idx], y[train_idx])
    val_pred = pipeline.predict_proba(X[val_idx])[:, 1]
    # Save pipeline
    pickle.dump(pipeline, f'pipeline_fold_{fold}.pkl')
```

**Benefits:**

1. ✅ **No Data Leakage**: Vectorizer fits only on training data per fold
2. ✅ **Reproducibility**: One file contains entire transformation + model
3. ✅ **Easy Deployment**: Load and predict in one step
4. ✅ **Modularity**: Easy to swap vectorizers or models

### Artifact Standards

**Every experiment saves:**

1. **Model Files**: `pipeline_fold_1.pkl` to `pipeline_fold_5.pkl` (or `.h5` for Keras, directories for Transformers)
2. **Metrics**: `metrics.json` with comprehensive performance data
3. **Feature Info**: `feature_names_all_folds.json` (for interpretability)
4. **Submission**: `{experiment_name}_submission.csv` (averaged fold predictions)
5. **Optuna Study** (if tuned): `optuna_study.pkl` + visualizations

**Detailed tracking maintained in**: `experiments/MODEL_TRACKING.md`

---

## Experiments Summary

### Experiment Progression

| #       | Experiment                  | Model Type                   | Vectorizer/Features                   | CV ROC-AUC      | Kaggle ROC-AUC | Notes                              |
| ------- | --------------------------- | ---------------------------- | ------------------------------------- | --------------- | -------------- | ---------------------------------- |
| **1**   | `exp_1_tfidf_log`           | Logistic Regression          | TF-IDF (10k features, 1-2gram)        | 0.9923 ± 0.0017 | 0.98954        | Baseline model                     |
| **1b**  | `exp_1_tfidf_log_optuna`    | Logistic Regression (Optuna) | TF-IDF (10k, 1-4gram)                 | 0.9950          | 0.99611        | Hyperparameter tuning (+50 trials) |
| **2**   | `exp_2_tfidf_ridge`         | Ridge Classifier             | TF-IDF (10k, 1-2gram)                 | 0.9965 ± 0.0010 | 0.99583        | Better regularization              |
| **3**   | `exp_3_struct_xgboost`      | XGBoost                      | 33 Structured Features Only           | 0.9768 ± 0.0049 | 0.97949        | Features alone insufficient        |
| **4**   | `exp_4_all_xgboost`         | XGBoost                      | TF-IDF (SVD 300) + Structured         | 0.9929 ± 0.0014 | 0.99736        | Combined text + features           |
| **5**   | `exp_5_all_mlp`             | MLP (3 layers)               | TF-IDF (SVD 300) + Structured         | 0.9962 ± 0.0018 | 0.99685        | Neural network approach            |
| **5b**  | `exp_5_all_mlp_optuna`      | MLP (Optuna)                 | TF-IDF (SVD 300) + Structured         | 0.9962 ± 0.0018 | **0.99850**    | Tuned architecture                 |
| **6**   | `exp_6_bilstm`              | BiLSTM (2 layers)            | Word embeddings (200 maxlen)          | 0.9933 ± 0.0032 | 0.99178        | Sequential modeling                |
| **7**   | `exp_7_charcnn`             | CharCNN                      | Character embeddings (1000 maxlen)    | 0.9848 ± 0.0044 | 0.99520        | Character-level features           |
| **8**   | `exp_8_hybrid_charcnn_mlp`  | Hybrid: CharCNN + MLP        | Char embeddings + TF-IDF + Structured | 0.9948 ± 0.0042 | **0.99886**    | Multi-branch architecture          |
| **9**   | `exp_9_hybrid_bilstm_mlp`   | Hybrid: BiLSTM + MLP         | Word embeddings + TF-IDF + Structured | 0.9956 ± 0.0023 | 0.99320        | Alternative hybrid                 |
| **10**  | `exp_10_stacked_ensemble`   | Stacked Ensemble             | Meta-learner on base models           | 0.9979 ± 0.0009 | **0.99893**    | Logistic meta-learner              |
| **10a** | `exp_10a_simple_ensemble`   | Simple Average               | Model averaging                       | -               | **0.99937**    | Weighted average                   |
| **10b** | `exp_10b_weighted_ensemble` | Weighted Ensemble            | CV-optimized weights                  | 0.9977          | **0.99933**    | Performance-based weights          |
| **11**  | `exp_11_roberta`            | RoBERTa-base                 | Transformer fine-tuning               | 0.9993 ± 0.0004 | -              | Pre-trained language model         |
| **12**  | `exp_12_deberta`            | DeBERTa-v3-base              | Transformer fine-tuning               | 0.9988 ± 0.0013 | **0.99964**    | **Best single model**              |

### Experiment Categories

**Traditional ML (Exp 1-4):**

- TF-IDF + Linear Models: Fast, interpretable baselines
- Gradient Boosting: Handles structured features well
- Optuna hyperparameter tuning for optimization

**Neural Networks (Exp 5-7):**

- MLP: Dense layers on dimensionality-reduced features
- BiLSTM: Sequential dependencies in word sequences
- CharCNN: Character-level pattern recognition

**Hybrid Architectures (Exp 8-9):**

- Multi-branch models combining deep learning + engineered features
- CharCNN/BiLSTM for text + MLP for structured data
- Fusion layers for feature integration

**Transformers (Exp 11-12):**

- Fine-tuned pre-trained models (RoBERTa, DeBERTa)
- Transfer learning from massive text corpora
- State-of-the-art performance

**Ensembles (Exp 10):**

- Simple averaging, weighted averaging, stacked generalization
- Combines strengths of diverse model families
- Consistent top-tier performance

---

## Key Machine Learning Techniques

### 1. Stratified K-Fold Cross-Validation

- **Purpose**: Maintain class distribution across folds
- **Implementation**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Benefit**: Reliable performance estimates for imbalanced data

### 2. Pipeline-Based Training (Data Leakage Prevention)

- **Purpose**: Ensure transformations fit only on training data
- **Implementation**: `sklearn.pipeline.Pipeline` with vectorizer + model
- **Benefit**: Eliminates data leakage in cross-validation

### 3. Hyperparameter Optimization (Optuna)

- **Purpose**: Automated hyperparameter search
- **Implementation**: Bayesian optimization with pruning
- **Configuration**:
  - Objective: Maximize ROC-AUC
  - Trials: 50-100 per experiment
  - Pruning: MedianPruner for early stopping
- **Experiments**: exp_1b, exp_4b, exp_5b, exp_3b
- **Tools**: `optuna.study`, visualization with plotly

### 4. Early Stopping (Neural Networks)

- **Purpose**: Prevent overfitting, reduce training time
- **Implementation**: `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
- **Applied in**: BiLSTM, CharCNN, Hybrid models, Transformers
- **Benefit**: Automatically stops when validation loss plateaus

### 5. Learning Rate Scheduling

- **Purpose**: Improve convergence and generalization

**Reduce LR on Plateau** (Keras models):

```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
```

**Linear Warmup + Decay** (Transformers):

```python
# DeBERTa, RoBERTa
get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Warmup Configuration**:

- Warmup ratio: 10% of training steps
- Linear decay after warmup phase
- Prevents unstable early training

### 6. Regularization Techniques

**Dropout** (Neural Networks):

- MLP: 0.3-0.5 dropout between dense layers
- BiLSTM: 0.2 recurrent dropout
- CharCNN: SpatialDropout1D(0.2) after embeddings
- Transformers: Built-in attention dropout

**Batch Normalization**:

- Applied in MLP and hybrid architectures
- Normalizes activations between layers
- Stabilizes training and improves convergence

**Weight Decay** (L2 Regularization):

- Transformers: 0.01 weight decay
- Prevents large weight values
- Improves generalization

### 7. Class Weighting

- **Purpose**: Handle class imbalance (35% AI, 65% Human)
- **Implementation**: `compute_class_weight('balanced', classes=[0,1], y=labels)`
- **Applied in**: Transformer models (DeBERTa, RoBERTa)
- **Effect**: Upweights minority class loss

### 8. Mixed Precision Training (FP16)

- **Purpose**: Faster training with reduced memory usage
- **Implementation**: `mixed_precision.Policy('mixed_float16')`
- **Applied in**: All GPU-accelerated models (Transformers, Hybrid NNs on Colab)
- **Hardware**: Leverages Tensor Cores on modern GPUs
- **Benefit**: ~2x speedup without accuracy loss

### 9. Gradient Accumulation (Transformers)

- **Purpose**: Simulate larger batch sizes with limited GPU memory
- **Configuration**:
  - Physical batch size: 16-32
  - Accumulation steps: 1-2
  - Effective batch size: 32-64
- **Benefit**: Stable training with large models

### 10. Gradient Clipping

- **Purpose**: Prevent exploding gradients
- **Implementation**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Applied in**: All RNN-based and Transformer models
- **Benefit**: Training stability

### 11. Dimensionality Reduction (TruncatedSVD)

- **Purpose**: Reduce TF-IDF feature space for tree-based and MLP models
- **Implementation**: `TruncatedSVD(n_components=300, random_state=42)`
- **Applied in**: XGBoost, MLP, Hybrid models
- **Benefit**: Faster training, reduced overfitting

### 12. Feature Scaling (StandardScaler)

- **Purpose**: Normalize structured features for neural networks
- **Implementation**: `StandardScaler()` within pipelines
- **Applied in**: MLP, Hybrid models
- **Benefit**: Improves neural network convergence

### 13. Model Checkpointing

- **Purpose**: Save best model during training
- **Implementation**: `ModelCheckpoint(monitor='val_roc_auc', save_best_only=True)`
- **Applied in**: All neural network experiments
- **Benefit**: Preserves best weights even if training diverges

### 14. Data Augmentation (Considered but Not Implemented)

- **Techniques explored**: Synonym replacement, back-translation
- **Decision**: Not implemented due to risk of label noise
- **Note**: Text augmentation for AI vs Human is challenging

### 15. Ensemble Diversity

- **Strategy**: Combine diverse model families
- **Base models**: Traditional ML + Neural Networks + Transformers
- **Methods**: Simple averaging, weighted averaging, stacked generalization
- **Benefit**: Reduces variance and improves robustness

---

## Best Performing Models

### Top 5 by Kaggle Score (Original Data)

| Rank | Model                   | Kaggle ROC-AUC | CV ROC-AUC      | Generalization Gap |
| ---- | ----------------------- | -------------- | --------------- | ------------------ |
| 🥇 1 | **DeBERTa-v3-base**     | **0.99964**    | 0.9994 ± 0.0002 | -0.00024           |
| 🥈 2 | Simple Average Ensemble | 0.99937        | -               | -                  |
| 🥉 3 | Weighted Ensemble (LL)  | 0.99933        | 0.9977          | -                  |
| 4    | Weighted Ensemble (AUC) | 0.99929        | 0.9977          | -                  |
| 5    | Stacked Ensemble        | 0.99893        | 0.9979 ± 0.0009 | 0.000              |

### Model Family Performance Summary

| Family                       | Best Kaggle | Avg CV ROC-AUC | Characteristics                                           |
| ---------------------------- | ----------- | -------------- | --------------------------------------------------------- |
| **Transformers**             | 0.99964     | 0.9991         | Pre-trained language understanding, contextual embeddings |
| **Ensembles**                | 0.99937     | 0.9977         | Combines diverse models, reduces variance                 |
| **Hybrid (DL + Features)**   | 0.99886     | 0.9952         | Multi-branch architectures, feature fusion                |
| **Neural Networks**          | 0.99850     | 0.9962         | Non-linear feature interactions                           |
| **Gradient Boosting**        | 0.99736     | 0.9859         | Tree-based, handles structured features                   |
| **Linear Models**            | 0.99611     | 0.9950         | Fast, interpretable, strong baselines                     |
| **Deep Learning (LSTM/CNN)** | 0.99520     | 0.9890         | Sequential/spatial patterns in text                       |

---

## Ensemble Methods

### 1. Simple Average Ensemble (`exp_10a`)

**Kaggle Score**: 0.99937

**Approach**:

- Equal-weight averaging of predictions from top models
- Models: DeBERTa, RoBERTa, Hybrid CharCNN+MLP, XGBoost, MLP

**Benefits**:

- Simple, no overfitting to validation data
- Robust to individual model failures
- Strong performance in practice

### 2. Weighted Ensemble (`exp_10b`)

**Kaggle Score**: 0.99933 (Log Loss optimized)

**Approach**:

- Optimize weights based on cross-validation performance
- Two variants: ROC-AUC optimized, Log Loss optimized
- Grid search or continuous optimization for weights

**Configuration**:

```json
{
  "deberta": 0.45,
  "hybrid_charcnn_mlp": 0.25,
  "xgboost_optuna": 0.15,
  "mlp_optuna": 0.1,
  "ridge": 0.05
}
```

**Benefits**:

- Upweights better-performing models
- Optimized for specific metric

### 3. Stacked Ensemble (`exp_10`)

**Kaggle Score**: 0.99893  
**CV ROC-AUC**: 0.9979 ± 0.0009

**Architecture**:

- **Level 0 (Base Models)**:
  - Hybrid CharCNN+MLP
  - XGBoost (Optuna-tuned)
  - Ridge Classifier
- **Level 1 (Meta-Learner)**:
  - Logistic Regression
  - Trained on out-of-fold predictions from base models

**Training Process**:

1. Generate OOF predictions for each base model (5-fold CV)
2. Concatenate OOF predictions as meta-features
3. Train meta-learner on these features with same CV splits
4. For test: average base model predictions, then apply meta-learner

**Benefits**:

- Learns optimal combination of models
- Can capture complex interactions between base predictions
- Generalizes well (shown by low CV std)

**Important**: All base models use `random_state=42`, ensuring fold alignment

### Ensemble Design Principles

✅ **Diversity**: Combine different model families (linear, tree, neural, transformer)  
✅ **Quality**: Only include high-performing base models (CV ROC-AUC > 0.99)  
✅ **Stability**: Prefer models with low cross-validation variance  
✅ **Fold Alignment**: Use identical CV splits (random_state=42) for stacking  
✅ **No Leakage**: Meta-learner trained on out-of-fold predictions only

---

## Lessons Learned

### What Worked Well

1. **Stratified K-Fold CV with Fixed Random State**

   - Consistent fold splits across all experiments enabled fair comparison
   - Essential for stacking and model selection

2. **Pipeline-Based Training**

   - Eliminated data leakage concerns
   - Made experiments reproducible and modular
   - Easy to swap components (vectorizers, models)

3. **Comprehensive Feature Engineering**

   - 33 structured features provided strong signal
   - Readability and entropy metrics particularly useful
   - Complemented text vectorization approaches

4. **Progressive Complexity**

   - Starting with baselines (Logistic Regression) established performance floor
   - Gradual increase in model complexity helped identify diminishing returns
   - Transformers provided final performance boost

5. **Hyperparameter Tuning (Optuna)**

   - Automated search found better configurations than manual tuning
   - Particularly effective for tree-based models (XGBoost)
   - Visualization tools helped understand parameter importance

6. **Early Stopping & LR Scheduling**

   - Prevented overfitting in neural networks
   - Reduced training time significantly
   - ReduceLROnPlateau improved convergence

7. **Ensemble Diversity**

   - Combining different model families (linear, tree, neural, transformer) reduced variance
   - Simple averaging often as good as complex stacking

8. **Mixed Precision Training (FP16)**
   - ~2x speedup for transformer training on Colab GPU
   - No accuracy degradation observed

### Challenges & Solutions

1. **Class Imbalance (35% AI, 65% Human)**

   - **Solution**: Stratified CV + class weights in loss function
   - **Alternative tried**: SMOTE (not effective for high-dimensional text)

2. **GPU Memory Constraints (Transformers)**

   - **Solution**: Gradient accumulation (effective batch size 32-64)
   - **Solution**: Mixed precision training (FP16)
   - **Solution**: Reduced sequence length (512 → 256 for RoBERTa)

3. **Data Leakage Risk**

   - **Solution**: Pipeline-based approach for traditional ML
   - **Solution**: Per-fold vectorizer fitting
   - **Solution**: Structured features computed independently per sample

4. **Overfitting in Complex Models**

   - **Solution**: Dropout (0.3-0.5), L2 regularization, BatchNorm
   - **Solution**: Early stopping with patience=2-3
   - **Solution**: Cross-validation to monitor generalization

5. **Long Training Times (Transformers)**

   - **Solution**: Early stopping (patience=2)
   - **Solution**: Mixed precision (FP16)
   - **Solution**: Used Colab Pro for T4/V100 GPUs
   - **Result**: DeBERTa training ~20 minutes for 5-fold CV

6. **Ensemble Fold Misalignment**
   - **Initial issue**: Different random states caused OOF mismatches
   - **Solution**: Standardized `random_state=42` across all experiments
   - **Verification**: Checked fold sizes and indices manually

### What Didn't Work

1. **POS Features in Deep Learning Models**

   - Added minimal value when transformers already capture syntax
   - Increased feature dimensionality without performance gain
   - **Conclusion**: Useful for traditional ML, not for transformers

2. **Character-Level Models Alone**

   - CharCNN performance (0.99520) lagged behind word-level models
   - Required very long sequences (maxlen=1000)
   - **Conclusion**: Better as hybrid component than standalone

3. **Aggressive Dimensionality Reduction**

   - TruncatedSVD with n_components < 200 hurt performance
   - Sweet spot: 300 components for most models
   - **Conclusion**: Balance between speed and information retention

4. **Complex Stacking Architectures**

   - Multi-level stacking (3+ levels) didn't improve over 2-level
   - Overfitting risk increased with complexity
   - **Conclusion**: Simpler ensembles (averaging, 2-level stacking) more robust

5. **Fine-Tuning All Transformer Layers**
   - Considered freezing lower layers to speed up training
   - **Result**: Minimal speedup, slightly worse performance
   - **Conclusion**: Full fine-tuning necessary for best results

### Key Insights

1. **Transformers are Worth the Compute Cost**

   - DeBERTa (+0.007 ROC-AUC over best non-transformer)
   - Pre-trained knowledge crucial for text understanding

2. **Feature Engineering Still Matters**

   - Hybrid models (text + structured features) outperformed text-only CNNs/LSTMs
   - Engineered features complementary to learned representations

3. **Cross-Validation Variance is Informative**

   - Low std models (transformers: ±0.0002) generalize better
   - High std models (CharCNN: ±0.0044) more sensitive to data splits

4. **Ensemble Gains are Modest but Consistent**

   - Simple ensemble: +0.0005 over best single model
   - Complex stacking: +0.0003 over simple ensemble
   - **Conclusion**: Ensembling provides small, reliable boost

5. **Deduplication Impact**
   - Deduplicated data: Cleaner training set, slightly different CV scores
   - Original data: More samples, but with duplicates
   - **Result**: Models trained on both; transformers robust to either

---

## Reproducibility

### Environment Setup

**Python Version**: 3.8+

**Core Libraries**:

```
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
xgboost==1.7.5
lightgbm==3.3.5
optuna==3.1.1
```

**Deep Learning**:

```
tensorflow==2.12.0
torch==2.0.1
transformers==4.28.1
accelerate==0.18.0
```

**NLP & Features**:

```
textstat==0.7.3
nltk==3.8.1
spacy==3.5.2
```

**Visualization**:

```
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1
```

### Random Seeds

**Consistent across all experiments**:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### Cross-Validation Split

**To reproduce exact folds**:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Train models here
```

### Running Experiments

**Prerequisites**:

1. Download competition data to `data/` folder
2. Run `Processing.ipynb` to generate structured features
3. Install dependencies: `pip install -r requirements.txt`

**Traditional ML (Fast)**:

- `1-Log-TFIDF.ipynb` - Logistic Regression baseline
- `2-Ridge-TFIDF.ipynb` - Ridge Classifier
- `3-XGB-STRUCT.ipynb` - XGBoost on structured features
- `4-XGB-ALL.ipynb` - XGBoost with text + features

**Neural Networks (Moderate Speed, GPU Recommended)**:

- `5-MLP-ALL.ipynb` - Multi-layer perceptron
- `6_BiLSTM.ipynb` - Bidirectional LSTM
- `7-CharCNN.ipynb` - Character-level CNN

**Hybrid Architectures (GPU Required)**:

- `8-hybridNN1.ipynb` - CharCNN + MLP fusion
- `9-hybridNN2.ipynb` - BiLSTM + MLP fusion

**Transformers (GPU Required, Colab Pro Recommended)**:

- `11_RoBERTa.ipynb` - RoBERTa-base fine-tuning (~25 min)
- `12_DeBERTa.ipynb` - DeBERTa-v3-base fine-tuning (~20 min)

**Ensembles (Requires Base Models)**:

- `10-Ensemble.ipynb` - Stacked ensemble
- `10-Ensemble2.ipynb` - Weighted average ensemble

**Evaluation**:

- `Evaluation.ipynb` - Comprehensive model comparison and analysis

### Hardware Recommendations

**Minimum (Traditional ML)**:

- CPU: 4 cores
- RAM: 8 GB
- Time: ~30 minutes for all traditional ML experiments

**Recommended (Neural Networks)**:

- GPU: NVIDIA GTX 1060 or better
- VRAM: 6 GB
- RAM: 16 GB
- Time: ~2 hours for all neural network experiments

**Optimal (Transformers)**:

- GPU: NVIDIA T4, V100, or A100 (Colab Pro)
- VRAM: 16 GB+
- RAM: 32 GB+
- Time: ~1 hour for both transformer experiments

### File Structure

```
ai-vs-human-text/
├── README.md                          # This file
├── EDA.ipynb                          # Exploratory data analysis
├── Processing.ipynb                   # Feature engineering pipeline
├── requirements.txt                   # Python dependencies
├── data/
│   ├── train.csv                      # Raw training data
│   ├── test.csv                       # Raw test data
│   ├── train_w_struct.csv             # Train + engineered features
│   └── test_w_struct.csv              # Test + engineered features
├── experiments/                       # Deduplicated data models
│   ├── MODEL_TRACKING.md              # Experiment standards
│   ├── exp_1_tfidf_log/               # Logistic Regression
│   ├── exp_2_tfidf_ridge/             # Ridge Classifier
│   ├── exp_4_all_xgboost/             # XGBoost
│   ├── exp_8_hybrid_charcnn_mlp/      # Hybrid CNN+MLP
│   ├── exp_10b_weighted_ensemble/     # Weighted ensemble
│   └── exp_12_deberta/                # DeBERTa (best model)
├── experiments_old/                   # Original data models (with duplicates)
│   ├── exp_1_tfidf_log/
│   ├── exp_5_all_mlp/
│   ├── exp_6_bilstm/
│   ├── exp_7_charcnn/
│   ├── exp_9_hybrid_bilstm_mlp/
│   ├── exp_10_stacked_ensemble/
│   ├── exp_11_roberta/
│   └── exp_12_deberta/
├── 1-Log-TFIDF.ipynb                  # Experiment notebooks
├── 2-Ridge-TFIDF.ipynb
├── 3-XGB-STRUCT.ipynb
├── 4-XGB-ALL.ipynb
├── 5-MLP-ALL.ipynb
├── 6_BiLSTM.ipynb
├── 7-CharCNN.ipynb
├── 8-hybridNN1.ipynb
├── 9-hybridNN2.ipynb
├── 10-Ensemble.ipynb
├── 11_RoBERTa.ipynb
├── 12_DeBERTa.ipynb
├── Evaluation.ipynb                   # Comprehensive evaluation
├── comprehensive_model_comparison.csv # All model results
├── kaggle_scores.csv                  # Leaderboard scores
└── model_family_summary.csv           # Performance by family
```

---

## Contact & Acknowledgments

**Course**: BT4012 - Fraud Analytics (NUS Business School)  
**Year**: 2025  
**Competition Platform**: Kaggle (In-Class Competition)

**Key Resources**:

- scikit-learn documentation
- Hugging Face Transformers
- Optuna documentation
- Kaggle forums and kernels

**Special Thanks**:

- Course instructors for providing the dataset and competition framework
- Hugging Face for pre-trained transformer models
- Kaggle community for best practices and insights

---

## Results

**Competition**: BT4012 In-class Kaggle Competition 2 [2025]
- **Participants**: 113
- **Metric**: ROC-AUC
- **Evaluation**: Final ranking based on private leaderboard (70% of test data)
- **My result**: 32nd place (private), dropped 13 places from public
- **Selected submission**: 0.99943 (deduplicated DeBERTa)
- **Best model (not selected)**: non-deduped DeBERTa at 0.99944 (one rank higher)
- **Gap left on table**: 0.00001

The public-to-private drop shrank from 53 places in Competition 1 to 13 here, and the model selection gap fell from 0.00139 to 0.00001. Better experiment tracking and more disciplined model selection directly caused that improvement. The public leaderboard was only 30% of the test data, so relying on CV instead of public scores was the right approach.

A few findings from the experiments:
- Local patterns were more discriminative than expected: unigram TF-IDF beat higher n-gram ranges, and CharCNN (character-level) beat BiLSTM (word-level)
- DeBERTa outperformed RoBERTa, probably because its disentangled attention separates positional from content-based attention, and content-based attention mattered more for this task
- The best single model (DeBERTa) outperformed all ensembles on the private set
- Modern transformers excel at text classification, but traditional ML plus feature engineering still works and is more interpretable

## Reflection

I applied what I learned from Competition 1. Experiment tracking was much more rigorous. Saving models, metrics, and out-of-fold predictions for each fold made ensembling significantly easier.

I discovered 500 duplicate samples (all human class) late in the process. Most experiments were run on the original dataset; only the final models were retrained on deduplicated data. Group stratified k-fold from the start would have been the better approach.

The leaderboard was only 30% of ~1,300 test samples, while CV used ~1,000 validation samples per fold. A small leaderboard is noisy, so rigorous CV with consistent folds is more reliable.

A full academic report is included as `report.pdf`.
