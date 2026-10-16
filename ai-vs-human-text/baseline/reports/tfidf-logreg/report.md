# Baseline evaluation: tfidf-word12-logreg

Dataset: competition. Created 2026-10-16T06:33:35Z.

Cross-validation on the training data. This is not a leaderboard score.

| Metric | Mean | Std (5 folds) |
|---|---|---|
| roc_auc | 0.9915 | 0.0018 |
| log_loss | 0.2482 | 0.0053 |
| accuracy | 0.9419 | 0.0076 |
| f1 | 0.9136 | 0.0122 |
| brier | 0.0622 | 0.0021 |

Rows: 4712 raw, 4162 unique after merging duplicates (550 removed, 0 conflicting texts dropped). 280 texts sit in 134 near-duplicate groups, each kept within one fold.
Test rows also in train: 230 of 1178 (181 unique texts).

Misclassified out-of-fold at 0.5: 242 (44 human texts scored as AI, 198 AI texts scored as human). Most confident errors:

- false positive (human text scored as AI), p(AI)=0.8304, fold 2: "This expansion will make our model more scalable and comprehensive, providing traders with valuable signals for a wider range of assets. With a broader perspective on the market, traders can make more informed decisions and seize opportunit"
- false positive (human text scored as AI), p(AI)=0.7302, fold 4: "In terms of future work, there are several strategies to enhance our predictive models. Firstly, further exploration of unsupervised clustering methods can be used for the identification of different types of fraud. By analysing abnormal co"
- false positive (human text scored as AI), p(AI)=0.6975, fold 0: "Git Sync was implemented for effective DAG management. This approach allows for automatic syncing of DAG files from a Git repository to the Kubernetes cluster, ensuring that the Directed Acyclic Graphs (DAGs) are always up-to-date with the "
- false positive (human text scored as AI), p(AI)=0.6907, fold 2: "To ensure the integrity and relevance of our dataset, we implemented several comprehensive pre-processing steps. These steps encompassed the removal of outliers, which could potentially skew our analysis, as well as filtering out null value"
- false positive (human text scored as AI), p(AI)=0.6709, fold 4: "Our hypothesis was that higher salaries lead to increased job satisfaction, as supported by our analysis. The outlier, Cloud Architect, initially skewed results, but after its removal, a clearer increasing trend emerged with a higher R2 val"
- false negative (AI text scored as human), p(AI)=0.1534, fold 0: "External service features are identified by querying various third-party services and search engines. For example, we determine if URLs are indexed by Google. If they are not indexed, it might suggest that the URLs are too new or short-live"
- false negative (AI text scored as human), p(AI)=0.1687, fold 1: "A strong linear relationship was observed between sheet count and speed loss, as reflected by an R-squared value of 0.99 when the sheet count was regressed against speed loss from the management report. This high R-squared value indicates a"
- false negative (AI text scored as human), p(AI)=0.1825, fold 0: "After implementing the recommendation system, we conducted a test using a question from the raw dataset to evaluate its performance. The question used for the test can be accessed at https://stackoverflow.com/questions/53992768, and a scree"
- false negative (AI text scored as human), p(AI)=0.2157, fold 0: "We generated image embeddings using the images from the training set. Every image was resized to a dimension of 224 x 224 with three color channels. To extract the image embeddings, we utilized a pre-trained ResNet50 model. This model trans"
- false negative (AI text scored as human), p(AI)=0.2179, fold 1: "According to the coherence score graph, we determined that the optimal number of topics is three. Below, you'll find a diagram showcasing the top 10 words for each topic. The first topic predominantly revolves around companies and their ope"
