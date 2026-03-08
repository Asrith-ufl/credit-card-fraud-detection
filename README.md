# Credit Card Fraud Detection

Comparing ML models on imbalanced transaction data to detect fraudulent credit card transactions.

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Class Imbalance:** 0.17% fraud (492 out of 284,807)

## Approach
1. Applied **SMOTE** to address class imbalance
2. Trained and compared 4 models:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Neural Network 
3. Evaluated using ROC-AUC, precision, recall, and F1-score

## Results

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.9708 |
| Random Forest | 0.9684 |
| **XGBoost** | **0.9800** |
| Neural Network | 0.9640 |

**Best Model:** XGBoost with 85% fraud recall and 0.98 AUC

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn
- Google Colab (GPU)
