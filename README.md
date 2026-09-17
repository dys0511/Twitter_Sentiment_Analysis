# Twitter Sentiment Analysis Pipeline

A modular machine learning project evaluating 5 algorithms on the Sentiment140 dataset using TF-IDF feature extraction.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **0.7329** | **0.7250** | 0.7504 | **0.7375** | 🥇 1st Place |
| **Random Forest** | 0.6969 | 0.6583 | 0.8188 | 0.7298 | 🥈 2nd Place |
| **XGBoost** | 0.6816 | 0.6361 | 0.8488 | 0.7272 | 🥉 3rd Place |
| **Gradient Boosting** | 0.6779 | 0.6337 | 0.8434 | 0.7236 | 4th Place |
| **AdaBoost** | 0.5999 | 0.5610 | **0.9186** | 0.6966 | 5th Place |

## Key Findings

* **Linear SVM** achieved the highest overall accuracy (73.29%) and F1-score (0.7375). Linear models excel with high-dimensional, sparse TF-IDF vectors.
* **Tree Ensembles** (Random Forest, XGBoost) tend to over-predict the positive class, resulting in high recall but lower precision.
