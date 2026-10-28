"""
Experiment 08: C4.5 Decision Tree Classification
Dataset: Breast Cancer Wisconsin Diagnostic (built into scikit-learn)

Important implementation note:
C4.5 uses gain ratio for split selection and handles continuous features.
scikit-learn's DecisionTreeClassifier(criterion="entropy") uses information
gain, not gain ratio, so it is NOT a true C4.5 implementation.
This experiment uses the open-source `chefboost` package with algorithm="C4.5".
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# C4.5 implementation package
try:
    from chefboost import Chefboost as chef
except ImportError as exc:
    raise SystemExit(
        "Missing package 'chefboost'. Install it in your Jupyter environment "
        "with: %pip install chefboost\n"
        "Then restart the kernel and run this program again."
    ) from exc

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
target_names = data.target_names

print("Dataset: Breast Cancer Wisconsin Diagnostic")
print("Dataset source: scikit-learn built-in dataset")
print(f"Number of rows: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("Class labels:", dict(enumerate(target_names)))

# Chefboost expects the target column to be the final column.
df = X.copy()
df["Decision"] = y.map({0: "malignant", 1: "benign"})

# 2. Split data; stratify preserves class proportions.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=y
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

# 3. Train C4.5 model using gain ratio.
config = {
    "algorithm": "C4.5",
    "enableParallelism": False
}
model = chef.fit(train_df, config=config, target_label="Decision")

# 4. Predict test samples.
# Chefboost's predict accepts a list of feature values in the same order as X.
feature_names = list(X.columns)
y_true = test_df["Decision"].tolist()
y_pred = []

for _, row in test_df[feature_names].iterrows():
    prediction = chef.predict(model, row.tolist())
    # Normalize common prediction return formats.
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()
    y_pred.append(str(prediction).strip())

# Normalize labels to avoid casing/whitespace differences.
y_true_norm = [str(v).strip().lower() for v in y_true]
y_pred_norm = [str(v).strip().lower() for v in y_pred]

# 5. Evaluate
accuracy = accuracy_score(y_true_norm, y_pred_norm)
labels = ["malignant", "benign"]

print("\nAlgorithm: C4.5 Decision Tree (gain ratio)")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(
    y_true_norm, y_pred_norm, labels=labels,
    target_names=labels, zero_division=0
))
cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)
print("Confusion Matrix (malignant, benign):")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
).plot(cmap="Blues", values_format="d")
plt.title("C4.5 Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.show()

# 6. Show a few sample predictions
print("\nSample predictions:")
for i in range(min(5, len(test_df))):
    print(
        f"Sample {i + 1}: Predicted = {y_pred[i]}, "
        f"Actual = {y_true_norm[i]}"
    )

print(
    "\nNote: Chefboost builds its own tree files when training. "
    "If the package/API version differs, check the Chefboost documentation "
    "or install it in the same environment as this notebook."
)
