"""
Experiment 07: Decision Tree Classification
Dataset: Breast Cancer Wisconsin Diagnostic (built into scikit-learn)
Rows: 569 | Features: 30
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

# 1. Load the built-in dataset (no Kaggle download required)
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
target_names = data.target_names

print("Dataset: Breast Cancer Wisconsin Diagnostic")
print("Dataset source: scikit-learn built-in dataset")
print(f"Number of rows: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("Class labels:", dict(enumerate(target_names)))
print("\nClass distribution:")
print(y.value_counts().sort_index().rename(index=dict(enumerate(target_names))))

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create and train the Decision Tree model
# max_depth limits complexity; random_state makes results reproducible.
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Predict test labels and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel: DecisionTreeClassifier(criterion='gini', max_depth=4)")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# 6. Visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=target_names, cmap="Blues", values_format="d"
)
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Plot the trained decision tree
plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=list(target_names),
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("Decision Tree Classifier (maximum depth = 4)")
plt.tight_layout()
plt.show()

# 8. Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Display feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importance.head(10).round(4))

importance.head(10).sort_values().plot(kind="barh", figsize=(8, 5))
plt.xlabel("Feature importance")
plt.title("Top 10 Important Features - Decision Tree")
plt.tight_layout()
plt.show()

# 10. Demonstrate a prediction for one test sample
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
print("\nSample prediction:")
print("Predicted class:", target_names[prediction])
print("Actual class:   ", target_names[y_test.iloc[0]])
