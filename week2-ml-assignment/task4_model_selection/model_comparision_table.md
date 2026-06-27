## Task 4 – Model Selection Strategy

### Comparison Table

| Criterion | Linear Regression | Logistic Regression | K-Nearest Neighbours (KNN) | Support Vector Machine (SVM) | Naive Bayes |
|---|---|---|---|---|---|
| **Problem Type** | Regression | Binary / Multi-class Classification | Classification (& Regression) | Classification (& Regression) | Classification |
| **Output** | Continuous numeric value | Probability → Class label | Class label (majority vote) | Class label (via hyperplane) | Class label (via probability) |
| **Assumption** | Linear relationship between X and y | Log-odds are linear in features | No parametric assumption | Data is linearly separable (linear kernel) | Features are conditionally independent |
| **Training Speed** | Very fast | Fast | No training (lazy learner) | Slow on large data (kernel SVM) | Very fast |
| **Prediction Speed** | Very fast | Very fast | Slow (computes distances at prediction) | Fast (after training) | Very fast |
| **Interpretability** | High (slope & intercept are meaningful) | High (coefficients show feature impact) | Low | Low | Medium |
| **Handles Non-linearity** | No | No (without feature engineering) | Yes (naturally) | Yes (with RBF / poly kernels) | No |
| **Handles Imbalanced Data** | N/A | Moderate | Poor | Moderate | Poor |
| **Sensitive to Scale** | Moderate | Yes | Yes (distance-based) | Yes | No |
| **Handles Missing Data** | No | No | No | No | No |
| **Key Hyperparameter** | None (OLS) | Regularization (C) | Number of neighbours (k) | Kernel type, C, gamma | Smoothing (alpha) |
| **Key Advantage** | Simple, fast, interpretable | Probabilistic output, fast | Flexible, no training phase | Effective in high dimensions | Works well with small data |
| **Key Limitation** | Only linear relationships | Assumes linearity; poor with complex boundaries | Slow at prediction; memory-intensive | Slow on large datasets | Independence assumption rarely holds |
| **Industry Application** | House price prediction, sales forecasting | Fraud detection, disease diagnosis | Recommendation systems, image recognition | Text classification, bioinformatics | Spam filtering, sentiment analysis |

### Decision Guide

```
Is the target variable continuous or discrete?
    │
    ├── Continuous → Linear Regression
    │
    └── Discrete (Classification)
            │
            ├── Need probability output?     → Logistic Regression / Naive Bayes
            ├── Small dataset, non-linear?   → KNN
            ├── High-dimensional text data?  → SVM / Naive Bayes
            └── Tabular, interpretability?   → Logistic Regression
```

---
