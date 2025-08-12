# prediction.py
# Task 5 - Disease Dataset Prediction using Decision Tree & Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# =========================
# SETTINGS
# =========================
DATA_FILE = "data/DiseaseDataset.csv"
OUTPUT_DIR = "outputs"

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv(DATA_FILE)
print("✅ Data loaded. Shape:", df.shape)
print(df.head())

# Identify target column
possible_targets = ['prognosis', 'disease', 'target', 'label', 'target ']
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    target_col = df.columns[-1]

print("\n🎯 Target column:", target_col)
print(df[target_col].value_counts())

# =========================
# STEP 2: Preprocessing
# =========================
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# One-hot encode categorical features if any
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print("\n🔄 Applied one-hot encoding on:", cat_cols)

# Encode target labels if they are not numeric
le = None
if y.dtype == 'object' or str(y.dtype).startswith('category'):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("📌 Encoded target classes:", list(le.classes_))

print("\n✅ Data ready. Features:", X.shape, "Target:", y.shape)

# =========================
# STEP 3: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("📊 Train size:", X_train.shape, "Test size:", X_test.shape)

# =========================
# STEP 4: Decision Tree Model
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

print("\n🌳 Decision Tree Results:")
print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Test accuracy :", accuracy_score(y_test, y_test_pred))
print("\nClassification report:\n", classification_report(y_test, y_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

# Shallow Decision Tree Visualization (max_depth=3)
shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow.fit(X_train, y_train)

plt.figure(figsize=(16, 10))
plot_tree(
    shallow,
    feature_names=X.columns,
    class_names=(le.classes_ if le is not None else None),
    filled=True,
    fontsize=8
)
plt.title("Decision Tree (max_depth=3)")
plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_maxdepth3.png"))
plt.close()

# =========================
# STEP 5: Overfitting Check - Depth vs Accuracy
# =========================
depths = list(range(1, min(21, X_train.shape[1] + 1)))
train_scores, test_scores = [], []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker='o', label='Train Accuracy')
plt.plot(depths, test_scores, marker='o', label='Test Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Depth vs Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "dt_depth_vs_acc.png"))
plt.close()

best_d = depths[np.argmax(test_scores)]
print("\n✅ Best max_depth:", best_d, "with Test Accuracy:", max(test_scores))

# Cross-validation for Decision Tree
dt_tuned = DecisionTreeClassifier(max_depth=best_d, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_dt = cross_val_score(dt_tuned, X, y, cv=cv)
print("CV Scores (Decision Tree):", cv_scores_dt)
print("Mean CV Accuracy:", cv_scores_dt.mean())

# =========================
# STEP 6: Random Forest Model
# =========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🌲 Random Forest Results:")
print("Train accuracy:", rf.score(X_train, y_train))
print("Test accuracy :", accuracy_score(y_test, y_pred_rf))
print("\nClassification report:\n", classification_report(y_test, y_pred_rf))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf, X, y, cv=cv)
print("CV Scores (Random Forest):", cv_scores_rf)
print("Mean CV Accuracy:", cv_scores_rf.mean())

# =========================
# STEP 7: Feature Importance
# =========================
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feat_importances.head(20).to_csv(os.path.join(OUTPUT_DIR, "feature_importances_top20.csv"))

# Plot Top Features
top_n = min(10, len(feat_importances))
plt.figure(figsize=(8, 5))
plt.bar(range(top_n), feat_importances.values[:top_n])
plt.xticks(range(top_n), feat_importances.index[:top_n], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Top Features (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_top_features.png"))
plt.close()

print("\n🏁 All done! Plots and results saved in:", OUTPUT_DIR)
