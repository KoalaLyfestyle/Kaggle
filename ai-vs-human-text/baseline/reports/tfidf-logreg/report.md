# Baseline evaluation: tfidf-word12-logreg

Dataset: competition. Created 2026-10-13T08:37:35Z.

Cross-validation on the training data. This is not a leaderboard score.

| Metric | Mean | Std (5 folds) |
|---|---|---|
| roc_auc | 0.9920 | 0.0026 |
| log_loss | 0.2470 | 0.0093 |
| accuracy | 0.9440 | 0.0130 |
| f1 | 0.9167 | 0.0206 |
| brier | 0.0616 | 0.0042 |

Rows: 4712 raw, 4162 unique after merging duplicates (550 removed, 0 conflicting texts dropped). 280 texts sit in 134 near-duplicate groups, each kept within one fold.
Test rows also in train: 230 of 1178 (181 unique texts).

Misclassified out-of-fold at 0.5: 233 (41 human texts scored as AI, 192 AI texts scored as human). Most confident errors:

- false positive (human text scored as AI), p(AI)=0.828, fold 1: "This expansion will make our model more scalable and comprehensive, providing traders with valuable signals for a wider range of assets. With a broader perspective on the market, traders can make more informed decisions and seize opportunit"
- false positive (human text scored as AI), p(AI)=0.8106, fold 1: "This expansion will make our model more scalable and comprehensive, providing traders with valuable signals for a wider range of assets. With a broader perspective on the market, traders can make more informed decisions and seize opportunit"
- false positive (human text scored as AI), p(AI)=0.7128, fold 1: "In terms of future work, there are several strategies to enhance our predictive models. Firstly, further exploration of unsupervised clustering methods can be used for the identification of different types of fraud. By analysing abnormal co"
- false positive (human text scored as AI), p(AI)=0.7026, fold 0: "Music and songs contain intricate features that hold significance for listeners. The pervasive consumption of songs by individuals, coupled with research underscoring its impact on daily lives, emphasize the need for careful consideration o"
- false positive (human text scored as AI), p(AI)=0.6888, fold 1: "1.The two sets of game recommendations, based on the user's gaming history and similar users' gaming history, undergo a comprehensive evaluation through manual Exploratory Data Analysis (EDA) in comparison to the games within the input user"
- false negative (AI text scored as human), p(AI)=0.1613, fold 3: "External service features are identified by querying various third-party services and search engines. For example, we determine if URLs are indexed by Google. If they are not indexed, it might suggest that the URLs are too new or short-live"
- false negative (AI text scored as human), p(AI)=0.1776, fold 4: "A strong linear relationship was observed between sheet count and speed loss, as reflected by an R-squared value of 0.99 when the sheet count was regressed against speed loss from the management report. This high R-squared value indicates a"
- false negative (AI text scored as human), p(AI)=0.1955, fold 0: "After implementing the recommendation system, we conducted a test using a question from the raw dataset to evaluate its performance. The question used for the test can be accessed at https://stackoverflow.com/questions/53992768, and a scree"
- false negative (AI text scored as human), p(AI)=0.213, fold 0: "The team used Anvil to develop the front-end interface for our model implementation. Anvil provides flexible hosting options, allowing components such as the model, data sets, and code to be hosted either on the client-side or on Anvil's se"
- false negative (AI text scored as human), p(AI)=0.2162, fold 0: "We generated image embeddings using the images from the training set. Every image was resized to a dimension of 224 x 224 with three color channels. To extract the image embeddings, we utilized a pre-trained ResNet50 model. This model trans"
