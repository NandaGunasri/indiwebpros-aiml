"""
Task 2 – Linear Regression from Scratch

Build a simple linear regression model from scratch using only Python and NumPy.
Avoid using scikit-learn and compute slope, intercept, predictions, and Mean Squared Error (MSE) manually.
"""

# ============================================================
# Task 2: Linear Regression from Scratch using NumPy
# Group: Versatile | Mohan Babu University
# ============================================================

import numpy as np

# ─────────────────────────────────────────────
# Step 1: Create a synthetic dataset
# Simulating: House Size (sq ft) vs Price (in lakhs)
# ─────────────────────────────────────────────
X = np.array([600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000], dtype=float)
y = np.array([30, 40, 50, 60, 75, 90, 100, 110, 130, 160], dtype=float)

n = len(X)  # Number of data points

# ─────────────────────────────────────────────
# Step 2: Compute Means
# ─────────────────────────────────────────────
mean_X = np.mean(X)
mean_y = np.mean(y)

print(f"Mean of X (House Size): {mean_X:.2f} sq ft")
print(f"Mean of y (Price):      ₹{mean_y:.2f} lakhs")

# ─────────────────────────────────────────────
# Step 3: Compute Slope (m) using Least Squares Formula
# m = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
# ─────────────────────────────────────────────
numerator   = np.sum((X - mean_X) * (y - mean_y))
denominator = np.sum((X - mean_X) ** 2)
slope       = numerator / denominator

print(f"\nSlope (m):     {slope:.6f}")

# ─────────────────────────────────────────────
# Step 4: Compute Intercept (b)
# b = ȳ - m * x̄
# ─────────────────────────────────────────────
intercept = mean_y - slope * mean_X

print(f"Intercept (b): {intercept:.6f}")

# ─────────────────────────────────────────────
# Step 5: Make Predictions
# ŷ = m * X + b
# ─────────────────────────────────────────────
y_pred = slope * X + intercept

print("\n--- Prediction Results ---")
print(f"{'House Size (sqft)':<22} {'Actual Price (₹L)':<22} {'Predicted Price (₹L)':<22}")
print("-" * 66)
for xi, yi, yp in zip(X, y, y_pred):
    print(f"{xi:<22.0f} {yi:<22.1f} {yp:<22.2f}")

# ─────────────────────────────────────────────
# Step 6: Compute Mean Squared Error (MSE)
# MSE = (1/n) × Σ(yi - ŷi)²
# ─────────────────────────────────────────────
mse = np.sum((y - y_pred) ** 2) / n
rmse = np.sqrt(mse)

print(f"\nMean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error:   {rmse:.4f} lakhs")

# ─────────────────────────────────────────────
# Step 7: Predict price for a new house (1600 sq ft)
# ─────────────────────────────────────────────
new_size = 1600
predicted_price = slope * new_size + intercept
print(f"\nPredicted price for {new_size} sq ft house: ₹{predicted_price:.2f} lakhs")

# ─────────────────────────────────────────────
# Optional: ASCII Visualization (Scatter Plot)
# ─────────────────────────────────────────────
print("\n--- ASCII Scatter: Actual vs Predicted ---")
print("Price")
print("(Lakhs)")
for i in range(len(X)):
    bar_actual    = "█" * max(0, int(round(y[i] / 5)))
    bar_predicted = "░" * max(0, int(round(y_pred[i] / 5)))
    print(f"{int(X[i]):>5} sqft | Actual: {bar_actual:<35} {y[i]:.0f}")
    print(f"         | Predicted: {bar_predicted:<35} {y_pred[i]:.1f}")
    print()

