## Task 1 – Model Evaluation Metrics

### 1.1 Accuracy

**Definition:** Accuracy is the ratio of correctly predicted samples (both positive and negative) to the total number of samples in the dataset.

**Formula:**

```
Accuracy = (True Positives + True Negatives) / Total Samples
```

**Importance:** Accuracy provides a quick, high-level view of how often a model gets its predictions right. It is the most intuitive metric and serves as the starting point for any evaluation.

**Real-World Example:** A spam email classifier that correctly identifies 950 out of 1,000 emails (both spam and legitimate) has an accuracy of 95%.

**Industries Using It:** E-commerce (product recommendation), education (student performance prediction), and cybersecurity (malware detection).

---

### 1.2 Precision

**Definition:** Precision measures the proportion of predicted positive cases that are actually positive. It answers the question: "Of all the items the model labeled as positive, how many truly are positive?"

**Formula:**

```
Precision = True Positives / (True Positives + False Positives)
```

**Importance:** Precision is critical when the cost of a false positive is high — for instance, when falsely labeling a legitimate transaction as fraud wastes resources and frustrates customers.

**Real-World Example:** In a fraud detection system, if the model flags 100 transactions and only 60 are actually fraudulent, the precision is 60%.

**Industries Using It:** Banking (fraud detection), legal (document classification), healthcare (cancer screening alerts).

---

### 1.3 Recall

**Definition:** Recall (also called Sensitivity or True Positive Rate) measures the proportion of actual positive cases that the model successfully identifies. It answers: "Of all the actual positives, how many did the model catch?"

**Formula:**

```
Recall = True Positives / (True Positives + False Negatives)
```

**Importance:** Recall is crucial when missing a positive case is more dangerous than raising a false alarm — such as failing to detect a disease.

**Real-World Example:** In a COVID-19 detection model, if there are 200 infected patients and the model correctly identifies 180 of them, recall is 90%.

**Industries Using It:** Healthcare (disease diagnosis), disaster management (fault detection), insurance (claim verification).

---

### 1.4 F1 Score

**Definition:** The F1 Score is the harmonic mean of Precision and Recall. It balances both metrics into a single value, especially useful when the dataset is imbalanced.

**Formula:**

```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Importance:** When there is a significant class imbalance (e.g., far more negatives than positives), accuracy alone is misleading. F1 Score gives a balanced view of the model's performance on the minority class.

**Real-World Example:** In a rare disease detection model where precision is 0.70 and recall is 0.65, the F1 Score is approximately 0.674 — providing a unified performance score.

**Industries Using It:** Healthcare (rare disease prediction), NLP (named entity recognition), security (intrusion detection).

---

### 1.5 AUC-ROC

**Definition:** AUC (Area Under Curve) – ROC (Receiver Operating Characteristic) measures the model's ability to distinguish between classes across all classification thresholds. The ROC curve plots True Positive Rate against False Positive Rate; the AUC is the area under this curve.

**Formula / Range:**

```
AUC = 0.5 → Random guessing
AUC = 1.0 → Perfect classifier
```

**Importance:** AUC-ROC is threshold-independent, making it suitable for comparing multiple models regardless of the decision boundary chosen.

**Real-World Example:** A model predicting loan default with AUC = 0.92 is considered highly reliable, as it correctly ranks defaulters above non-defaulters 92% of the time.

**Industries Using It:** Banking (credit scoring), healthcare (patient risk stratification), marketing (churn prediction).

---

### 1.6 Deployment Decision – Hospital Scenario

**Given Metrics:**
- Accuracy: 99%
- Precision: 45%
- Recall: 38%

**Should this model be deployed in a hospital? Answer: No.**

Despite its impressive 99% accuracy, this model should **not** be deployed in a clinical environment. The high accuracy is misleading — it likely results from extreme class imbalance (e.g., most patients are healthy, so predicting "no disease" for everyone inflates accuracy). However, a precision of only 45% means that more than half of patients flagged as sick are actually healthy, causing unnecessary panic and resource waste. More critically, a recall of only 38% means the model misses 62% of actual patients who are ill. In a hospital setting, a missed diagnosis (false negative) can be life-threatening. Medical AI systems must prioritize recall above all else. This model requires significant improvement in both precision and recall before any clinical consideration.

---
