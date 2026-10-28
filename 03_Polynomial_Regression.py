# Experiment 3: Polynomial Regression (degree 2)
# Dataset: California Housing (20,640 observations; fetched by scikit-learn)
# First run requires an internet connection to download the dataset.
# PolynomialFeatures creates squared terms and pairwise interactions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nTarget: MedHouseVal (median house value, units of $100,000)")

# 2. Separate input features and target
X = df[housing.feature_names]
y = df["MedHouseVal"]

# 3. Split data (same split as linear regression for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

# 4. Create and train degree-2 polynomial regression model
degree = 2
model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate model
print(f"\n--- Polynomial Regression (degree {degree}) Test Evaluation ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# 7. Report number of polynomial features
poly = model.named_steps["polynomialfeatures"]
print("\nOriginal input features:", X.shape[1])
print("Polynomial features after expansion:", len(poly.get_feature_names_out(X.columns)))

# 8. Visualization: actual vs predicted values
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.25, s=12)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Polynomial Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Visualization: residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.25, s=12)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Median House Value")
plt.ylabel("Residual (actual - predicted)")
plt.title("Polynomial Regression: Residual Plot")
plt.tight_layout()
plt.show()

# 10. Visualize a one-feature relationship for interpretation:
# Median income vs median house value, with a quadratic fit.
# This plot is illustrative; the evaluated model above uses all 8 features.
income = df[["MedInc"]]
target = df["MedHouseVal"]
income_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
income_model.fit(income, target)
income_grid = pd.DataFrame({
    "MedInc": np.linspace(income["MedInc"].min(), income["MedInc"].max(), 300)
})
target_grid = income_model.predict(income_grid)

plt.figure(figsize=(8, 5))
plt.scatter(income["MedInc"], target, alpha=0.12, s=8, label="All observations")
plt.plot(income_grid["MedInc"], target_grid, color="red",
         linewidth=2, label="Quadratic fit using MedInc only")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value (units of $100,000)")
plt.title("Illustration of a Polynomial Relationship")
plt.legend()
plt.tight_layout()
plt.show()

# 11. Example prediction for one test observation
sample = X_test.iloc[[0]]
sample_prediction = model.predict(sample)[0]
print("\nExample input row:")
print(sample.to_string(index=False))
print("Predicted median house value:", round(sample_prediction, 4),
      "units of $100,000")
print("Approximate predicted value in dollars:", f"${sample_prediction * 100000:,.0f}")
