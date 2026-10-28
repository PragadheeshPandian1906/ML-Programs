"""
Experiment 09: CART Decision Tree Classification
Dataset: Breast Cancer Wisconsin Diagnostic (built into scikit-learn)

CART = Classification and Regression Trees.
For classification, this experiment uses Gini impurity and binary splits.
scikit-learn's DecisionTreeClassifier implements an optimized CART-style tree.
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# 1. Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
target_names = data.target_names

print("Dataset: Breast Cancer Wisconsin Diagnostic")
print("Dataset source: scikit-learn built-in dataset")
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])
print("Class labels:", dict(enumerate(target_names)))

print("\nClass distribution:")
print(
    y.value_counts()
     .sort_index()
     .rename(index=dict(enumerate(target_names)))
)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create the CART classifier.
# CART uses binary splits; Gini impurity is the default classification criterion.
model = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=4,
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAlgorithm: CART Decision Tree")
print("Criterion: Gini impurity")
print("Split strategy: Best binary split")
print("Maximum depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"ROC-AUC: {auc:.4f}")

# 7. Confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=target_names,
    cmap="Blues",
    values_format="d"
)
plt.title("CART - Confusion Matrix")
plt.tight_layout()
plt.show()

# 8. Plot the CART tree
plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=list(target_names),
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("CART Decision Tree (Maximum Depth = 4)")
plt.tight_layout()
plt.show()

# 9. ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"CART (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CART")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 10. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(importance.head(10).round(4))

importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(8, 5)
)
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features - CART")
plt.tight_layout()
plt.show()

# 11. Predict a single test sample
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]

print("\nSample Prediction:")
print("Predicted class:", target_names[prediction])
print("Actual class:   ", target_names[y_test.iloc[0]])
