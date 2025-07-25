# 📊 Customer Churn Prediction and Analysis Using Data Mining

This project predicts customer churn using data mining techniques and various machine learning models. It aims to identify customers at risk of leaving a telecom service provider, enabling proactive customer retention strategies.

---

## 🧠 Problem Statement

Customer churn directly impacts a company's revenue. Predicting churn helps in developing strategic retention campaigns. The goal is to identify key indicators of churn and build robust predictive models.

---

## 🧪 Solution Approach

Using data mining and machine learning, this project applies the following:

- Data Cleaning & Preprocessing
- Feature Engineering & Encoding
- Model Training & Comparison
- Model Evaluation & Interpretability
- Streamlit Integration for Live Predictions

---

## 📦 Dataset

- **Source:** IBM Telco Customer Churn Dataset  
- **Size:** 69,906 rows and 21 columns, resulting in a total of 1,468,026 values (69,906 rows * 21 columns)
- **Target Variable:** `Churn` (Yes/No)  
- **Features:** Customer account data, service subscriptions, usage patterns

---

## ⚙️ Key Steps

- ✅ Data Cleaning (`null`, `object` types)
- ✅ Label Encoding (binary) & One-Hot Encoding (categorical)
- ✅ Feature Scaling using `StandardScaler`
- ✅ Model Training with 7 ML Models
- ✅ Confusion Matrices & ROC, PR Curve Visualizations
- ✅ SHAP & Permutation Feature Importance
- ✅ Exporting final model for reuse (`.pkl`)
- ✅ Streamlit app for real-time predictions

---

## 🧰 Techniques Used

- Logistic Regression
- Decision Tree
- Random Forest ✅
- Tuned Random Forest (GridSearchCV)
- XGBoost
- LightGBM
- CatBoost

---

## 📊 Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  | Training Time (s) |
|--------------------|----------|-----------|--------|----------|------|--------------------|
| Logistic Regression| 80.43%   | 78.2%     | 75.6%  | 76.9%    | 0.84 | ~1.2               |
| Decision Tree      | 99.72%   | 99.6%     | 99.8%  | 99.7%    | 0.99 | ~0.9               |
| Random Forest      | 99.72%   | 99.7%     | 99.8%  | 99.7%    | 0.99 | ~12.5              |
| Tuned Random Forest| 98.72%   | 97.4%     | 98.1%  | 97.7%    | 0.98 | ~22.4              |
| XGBoost            | 93.57%   | 91.2%     | 90.5%  | 90.8%    | 0.95 | ~18.1              |
| LightGBM           | 85.77%   | 83.5%     | 82.6%  | 83.0%    | 0.88 | ~7.4               |
| CatBoost           | 91.52%   | 90.1%     | 89.3%  | 89.7%    | 0.94 | ~10.3              |




---

## ✅ Best Model: Tuned Random Forest

- **Accuracy:** 99.72%
- **AUC Score:** 0.99
- **F1 Score:** 0.99
- Saved to: `models/random_forest_model.pkl`

---

## 📈 Model Interpretability

- **Confusion Matrices (TP, FP, TN, FN)** for each model
- **ROC Curve Overlay** for all 7 models
- **Precision-Recall Curves** for top 3
- **SHAP Beeswarm Plot** for XGBoost
- **Permutation vs Built-in Feature Importance**

---

## 💡 How to Predict New Customer Churn

Use the function below from `03-predict-new-customer.ipynb`:

```python
predict_churn(new_customer_dict)
