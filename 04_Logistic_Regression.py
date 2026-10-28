# Experiment 4: Logistic Regression
# Dataset: Breast Cancer Wisconsin Diagnostic dataset (569 observations)
# Source: scikit-learn built-in dataset; no manual Kaggle download required.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

# Step 1: Load dataset
dataset = load_breast_cancer(as_frame=True)
df = dataset.frame

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nClass distribution:")
print(df["target"].map({0: "malignant", 1: "benign"}).value_counts())
print("\nMissing values:", int(df.isnull().sum().sum()))

# Step 2: Separate features and target
X = df[dataset.feature_names]
y = df["target"]  # 0 = malignant, 1 = benign

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

# Step 4: Create pipeline and train logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluate model
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=dataset.target_names, zero_division=0
))

# Step 7: Visualize class distribution
class_counts = df["target"].map({0: "malignant", 1: "benign"}).value_counts()
plt.figure(figsize=(7, 4))
class_counts.plot(kind="bar", edgecolor="black")
plt.xlabel("Diagnosis")
plt.ylabel("Number of cases")
plt.title("Breast Cancer Dataset: Class Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 8: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=dataset.target_names
).plot(values_format="d")
plt.title("Logistic Regression: Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 9: ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Step 10: Predict one sample
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
predicted_label = dataset.target_names[prediction]

print("\nExample input (first test sample):")
print(sample.to_string(index=False))
print("Predicted diagnosis:", predicted_label)
print("Probability of malignant:", f"{probabilities[0]:.4f}")
print("Probability of benign:", f"{probabilities[1]:.4f}")
