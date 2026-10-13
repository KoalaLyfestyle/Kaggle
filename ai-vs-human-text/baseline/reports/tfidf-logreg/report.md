# Baseline evaluation: tfidf-word12-logreg

Dataset: competition. Created 2026-10-13T08:29:56Z.

| Metric | Mean | Std (5 folds) |
|---|---|---|
| roc_auc | 0.9926 | 0.0015 |
| log_loss | 0.2445 | 0.0060 |
| accuracy | 0.9430 | 0.0133 |
| f1 | 0.9138 | 0.0216 |
| brier | 0.0609 | 0.0031 |

Rows: 4712 raw, 4212 unique after merging duplicates (500 removed, 0 conflicting texts dropped).
Test rows also in train: 212 of 1178 (166 unique texts).

Misclassified out-of-fold at 0.5: 240 (39 human texts scored as AI, 201 AI texts scored as human). Most confident errors:

- false positive (human text scored as AI), p(AI)=0.7192, fold 1: "This expansion will make our model more scalable and comprehensive, providing traders with valuable signals for a wider range of assets. With a broader perspective on the market, traders can make more informed decisions and seize opportunit"
- false positive (human text scored as AI), p(AI)=0.7139, fold 3: "In terms of future work, there are several strategies to enhance our predictive models. Firstly, further exploration of unsupervised clustering methods can be used for the identification of different types of fraud. By analysing abnormal co"
- false positive (human text scored as AI), p(AI)=0.6938, fold 1: "Git Sync was implemented for effective DAG management. This approach allows for automatic syncing of DAG files from a Git repository to the Kubernetes cluster, ensuring that the Directed Acyclic Graphs (DAGs) are always up-to-date with the "
- false positive (human text scored as AI), p(AI)=0.6923, fold 3: "Music and songs contain intricate features that hold significance for listeners. The pervasive consumption of songs by individuals, coupled with research underscoring its impact on daily lives, emphasize the need for careful consideration o"
- false positive (human text scored as AI), p(AI)=0.6835, fold 0: "A business can implement our comprehensive solution to enhance its customer's experience and drive growth. By developing a fake review detection model, the business can maintain the credibility of its review ecosystem. With sentiment analys"
- false negative (AI text scored as human), p(AI)=0.1658, fold 2: "A strong linear relationship was observed between sheet count and speed loss, as reflected by an R-squared value of 0.99 when the sheet count was regressed against speed loss from the management report. This high R-squared value indicates a"
- false negative (AI text scored as human), p(AI)=0.1737, fold 3: "External service features are identified by querying various third-party services and search engines. For example, we determine if URLs are indexed by Google. If they are not indexed, it might suggest that the URLs are too new or short-live"
- false negative (AI text scored as human), p(AI)=0.1758, fold 0: "After implementing the recommendation system, we conducted a test using a question from the raw dataset to evaluate its performance. The question used for the test can be accessed at https://stackoverflow.com/questions/53992768, and a scree"
- false negative (AI text scored as human), p(AI)=0.2071, fold 3: "We generated image embeddings using the images from the training set. Every image was resized to a dimension of 224 x 224 with three color channels. To extract the image embeddings, we utilized a pre-trained ResNet50 model. This model trans"
- false negative (AI text scored as human), p(AI)=0.214, fold 4: "According to the coherence score graph, we determined that the optimal number of topics is three. Below, you'll find a diagram showcasing the top 10 words for each topic. The first topic predominantly revolves around companies and their ope"
