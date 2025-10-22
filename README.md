# Customer Retention Intelligence System

End-to-end machine learning project to predict customer churn for e-commerce platforms. This system identifies customers at high risk of leaving, enabling targeted retention strategies to improve user retention and revenue.

---

## Tech Stack

**Programming & Data Handling:**  
- Python, Pandas, NumPy  

**Machine Learning & Modeling:**  
- Scikit-learn (Logistic Regression, SVM, Decision Tree, Random Forest)  
- XGBoost  

**Web App / Deployment:**  
- Streamlit  

---

## Project Preview



---

## Project Overview

- **Objective:** Predict whether a customer will churn based on historical behavioral and transactional data  
- **Dataset:** Features include Tenure, CityTier, PreferredPaymentMode, NumberOfDeviceRegistered, SatisfactionScore, CouponUsed, OrderCount, DaySinceLastOrder, and more  
- **Problem Type:** Binary Classification  
- **Challenge:** Class imbalance (~83% non-churn, ~17% churn)  

---

## Data Preprocessing

| Step | Description |
|------|-------------|
| **Categorical Handling** | One-Hot Encoding for linear models (Logistic Regression, SVM); Label Encoding for tree-based models (Decision Tree, Random Forest, XGBoost) |
| **Numerical Outliers** | Capped at 1st and 99th percentiles to remove extreme outliers |
| **Scaling** | StandardScaler applied to numerical columns for linear models |
| **Train/Test Split** | 80:20 split with stratify=y to maintain class distribution |

---

## Class Imbalance Handling

- Balanced class weights for linear models  
- SMOTE tested but improvement negligible (~0.01–0.1), so final models relied on class weights

---

## Models Implemented

| Model | Features | Key Hyperparameters | Test Performance |
|-------|----------|-------------------|----------------|
| Logistic Regression | One-Hot Encoded | class_weight='balanced' | Accuracy: 0.79, ROC-AUC: 0.886 |
| SVM (Linear) | One-Hot Encoded | class_weight='balanced' | Accuracy: 0.80, ROC-AUC: 0.883 |
| Decision Tree | Label Encoded | max_depth=None, class_weight='balanced' | Accuracy: 0.96, ROC-AUC: 0.92 |
| Random Forest | Label Encoded | n_estimators=200, class_weight='balanced' | Accuracy: 0.98, ROC-AUC: 0.999 |
| XGBoost | Label Encoded | n_estimators=200, scale_pos_weight | Accuracy: 0.99, ROC-AUC: 0.999 |

- Evaluated using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices  
- 5-fold cross-validation confirms high ROC-AUC  

---

## Feature Importance & Interpretability

- **SHAP (SHapley Additive exPlanations)** used in the Jupyter notebooks for tree-based models
- Single-customer interactive SHAP plots are not included in the Streamlit app to keep the app lightweight  
- Explains contribution of each feature to churn prediction  
- **Top features contributing to churn:**  
  - NumberOfDeviceRegistered  
  - SatisfactionScore  
  - CouponUsed  
  - OrderCount  
  - DaySinceLastOrder  
- Insights help design targeted retention campaigns for high-risk users  

---

## Key Findings

- Linear models provide reasonable baseline performance (~0.79 accuracy)  
- Tree-based models significantly outperform linear models  
- Random Forest and XGBoost are highly accurate and robust to class imbalance without SMOTE  
- Business impact: Retaining high-risk customers increases revenue and reduces churn  

---

## Repository Structure
```
customer-retention-system/
│
├─ data/                       # datasets
├─ Notebooks/                   # workflow with preprocessing, modeling, evaluation and SHAP analysis
├─ models/                      # Saved models and preprocessing objects
│   ├─ random_forest_model.pkl
│   ├─ xgboost_model.pkl
│   ├─ label_encoders.pkl
│   └─ ...                     
├─ deployment/
│   └─ app.py                   # Streamlit app
├─ .gitignore
├─ requirements.txt             # Dependencies
└─ README.md                    # Documentation
```

## How to Clone and Run the Repository

1. Clone the repository:

```bash
git clone https://github.com/Sravya-T01/customer-retention-system.git
```

2. Navigate to the project folder:
```bash
cd customer-retention-system
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run deployment/app.py
```

## Future Improvements

- Multi-customer batch predictions
- Interactive SHAP plots for single-customer insights
- Incorporate real-time transactional data for live predictions
- Extend model selection to other tree-based algorithms