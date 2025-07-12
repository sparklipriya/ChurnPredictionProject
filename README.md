# Customer Churn Prediction
This project predicts customer churn using data mining techniques.

# Customer Churn Prediction Using Data Mining

## Problem Statement
Customer churn reduces business revenue. Predicting churn allows proactive retention strategies.

## Solution
Using data mining and ML models (Logistic Regression, Decision Tree, Random Forest), we predict customer churn based on telecom usage data.

## Dataset
Source: Telco Customer Churn Dataset

## Key Steps:
- Data cleaning and preprocessing
- Feature encoding and engineering
- Model training and comparison
- Final churn prediction function

##  Techniques Used
- Data Cleaning & Handling Missing Values
- Binary & Categorical Encoding
- Feature Engineering
- Logistic Regression
- Decision Tree
- Random Forest ✅ (Best Accuracy)
- Model Persistence using `joblib`
- Real-Time Churn Prediction Function
  

## Evaluation
| Model              | Accuracy (%) |
|-------------------|--------------|
| Logistic Regression | ~79%        |
| Decision Tree       | ~81%        |
| **Random Forest**   | **~83–85%**  |

## Best Model: Random Forest (accuracy: ~82%)

## How to Predict New Churn:
Use `predict_churn(new_customer_dict)` from `03-predict-new-customer.ipynb`

