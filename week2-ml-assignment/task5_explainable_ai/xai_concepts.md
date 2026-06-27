## Task 5 – Explainable AI (XAI)

### 5.1 What is Explainable AI?

Explainable AI (XAI) refers to methods and techniques that make the decisions of machine learning models understandable to humans. As AI systems are deployed in high-stakes environments such as healthcare, finance, and legal systems, the ability to explain *why* a model made a specific prediction has become as important as the prediction itself. XAI bridges the gap between model accuracy and human trust.

### 5.2 Feature Importance

Feature importance quantifies the contribution of each input feature to the model's predictions. In tree-based models (such as Random Forest or XGBoost), it is computed based on how frequently and effectively each feature is used to split nodes. Features with high importance are the primary drivers of the model's behavior.

**Example:** In a loan default model, feature importance might reveal that *credit score* contributes 42% to predictions, followed by *income* at 28%. This tells risk managers which financial factors the model relies on most.

**Limitation:** Global feature importance shows average impact across all predictions but cannot explain individual decisions.

### 5.3 SHAP Values

SHAP (SHapley Additive exPlanations) values are grounded in cooperative game theory. They assign each feature a contribution score for a specific prediction by calculating its marginal impact across all possible feature combinations. Unlike global importance, SHAP values are **local** — they explain why the model made a particular prediction for a particular individual.

**Formula concept:**

```
Prediction = Base value + SHAP(feature_1) + SHAP(feature_2) + ... + SHAP(feature_n)
```

**Example:** For a patient classified as high-risk for heart disease, SHAP might show:
- Age (+0.35): increases risk
- Cholesterol (+0.28): increases risk
- Exercise Frequency (−0.22): reduces risk

This tells a doctor exactly what drove the AI's recommendation for that specific patient.

### 5.4 Why Banks and Hospitals Require Explainable Predictions

**Regulatory Compliance:** In India, the RBI mandates that credit decisions must be explainable to customers. In the EU, GDPR's "right to explanation" legally requires that automated decisions affecting individuals be interpretable. Hospitals are similarly governed by medical ethics frameworks requiring that AI-assisted diagnoses be reviewable by clinicians.

**Accountability and Trust:** When a bank rejects a loan or a hospital AI flags a patient for surgery, those affected individuals have the right to know why. Opaque "black box" models undermine trust and accountability. Explainable models allow clinicians, auditors, and customers to validate, override, or challenge AI outputs.

**Error Detection:** In critical systems, unexplained predictions can hide dangerous biases. A hospital AI that denies treatment to certain demographic groups due to biased training data would only be caught through XAI analysis.

**Real-World Example:** In 2019, a healthcare algorithm in the United States was found to systematically underserve Black patients. An explainability audit of the model's features revealed that it was using healthcare cost as a proxy for health need — a deeply biased assumption. SHAP analysis helped identify and correct this problem, preventing further harm.

---
