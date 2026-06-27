"""
Task 3 – Classification Model

Predict whether a bank customer will default on a loan based on their financial profile.

Dataset:
- Annual Income (₹): Customer's yearly income
- Loan Amount (₹): Amount of loan applied for
- Credit Score: Score between 300–850
- Employment Years: Number of years employed
- Existing Debt (₹): Current outstanding debt

Target Variable: Loan Default — 1 (Will Default), 0 (Will Not Default)

Algorithm Chosen: Logistic Regression
- Good for binary classification
- Interpretable and efficient
- Works well for linearly separable financial risk data
"""

# ============================================================
# Task 3: Loan Default Classification using Logistic Regression
# Group: Versatile | Mohan Babu University
# ============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Step 1: Create Synthetic Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 200

# Features
annual_income   = np.random.randint(200000, 1200000, n_samples)
loan_amount     = np.random.randint(50000, 800000, n_samples)
credit_score    = np.random.randint(300, 850, n_samples)
employment_yrs  = np.random.randint(0, 30, n_samples)
existing_debt   = np.random.randint(0, 500000, n_samples)

X = np.column_stack([annual_income, loan_amount,
                     credit_score, employment_yrs, existing_debt])

# Target: default more likely if low income, high loan, low credit
default_prob = (
    0.0000003 * (800000 - annual_income) +
    0.0000004 * loan_amount +
    0.002     * (600 - credit_score) +
    0.01      * (10 - employment_yrs) +
    0.0000002 * existing_debt
)
default_prob = np.clip(default_prob, 0, 1)
y = (np.random.rand(n_samples) < default_prob).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution → No Default: {sum(y==0)}, Default: {sum(y==1)}")

# ─────────────────────────────────────────────
# Step 2: Train-Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# Step 3: Feature Scaling
# Logistic Regression is sensitive to feature scale
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Step 4: Train the Model
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# Step 5: Predict and Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")

print("\n--- Confusion Matrix ---")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred,
      target_names=["No Default", "Default"]))

# ─────────────────────────────────────────────
# Step 6: Feature Importance (Coefficients)
# ─────────────────────────────────────────────
feature_names = ["Annual Income", "Loan Amount", "Credit Score",
                 "Employment Yrs", "Existing Debt"]
coefficients  = model.coef_[0]

print("--- Feature Coefficients (Impact on Default) ---")
for name, coef in zip(feature_names, coefficients):
    direction = "↑ Increases Risk" if coef > 0 else "↓ Reduces Risk"
    print(f"  {name:<20}: {coef:+.4f}  {direction}")

"""
Output Explanation:

The model is trained on a synthetically generated dataset of 200 loan applicants with features that realistically simulate loan default behavior - customers with low income, high loan amounts, poor credit scores, and existing debt are more likely to default.

After an 80/20 train-test split and standard scaling (necessary since logistic regression is scale-sensitive), the model learns a decision boundary separating defaulters from non-defaulters.

The confusion matrix breaks down predictions into four categories: True Positives (correctly caught defaulters), True Negatives (correctly cleared customers), False Positives (wrongly flagged), and False Negatives (missed defaulters). The classification report provides per-class precision and recall.

Feature coefficients reveal the model's reasoning - credit score and annual income carry negative coefficients (higher values reduce default risk), while loan amount and existing debt carry positive coefficients (they increase risk). This interpretability is a key advantage of logistic regression in regulated sectors like banking.

Challenges Faced:
- Class imbalance: fewer defaulters than non-defaulters, mitigated through stratified splitting.
- Feature scaling: raw rupee values differ in magnitude, addressed with StandardScaler.
- Threshold tuning: the default 0.5 threshold may not be optimal; further ROC analysis would refine this.
"""

