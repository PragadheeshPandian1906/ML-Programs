# Experiment 2: Multiple Linear Regression
# Dataset: California Housing (20,640 observations; fetched by scikit-learn)
# First run requires an internet connection to download the dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:", int(df.isnull().sum().sum()))
print("\nFeatures:", list(housing.feature_names))
print("Target: MedHouseVal (median house value, units of $100,000)")

# 2. Separate input features and target
X = df[housing.feature_names]
y = df["MedHouseVal"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

# 4. Build and train the model
# Scaling is included in the pipeline to keep preprocessing organized.
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate model
print("\n--- Test Set Evaluation ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# 7. Visualization: target distribution
plt.figure(figsize=(8, 5))
plt.hist(y, bins=40, edgecolor="black")
plt.xlabel("Median House Value (units of $100,000)")
plt.ylabel("Number of districts")
plt.title("Distribution of California House Values")
plt.tight_layout()
plt.show()

# 8. Visualization: actual vs predicted values
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.25, s=12)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Linear Regression: Actual vs Predicted")
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
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.show()

# 10. Display coefficients and a sample prediction
coefficients = pd.Series(
    model.named_steps["linearregression"].coef_,
    index=housing.feature_names
).sort_values(key=abs, ascending=False)
print("\nCoefficients (standardized input features; sorted by absolute value):")
print(coefficients.round(4))

sample = X_test.iloc[[0]]
sample_prediction = model.predict(sample)[0]
print("\nExample input row:")
print(sample.to_string(index=False))
print("Predicted median house value:", round(sample_prediction, 4),
      "units of $100,000")
print("Approximate predicted value in dollars:", f"${sample_prediction * 100000:,.0f}")
