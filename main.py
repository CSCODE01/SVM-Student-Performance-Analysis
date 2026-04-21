import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

data = pd.read_csv('Student_Performance.csv')

# Critical Fix: Remove any hidden spaces from column names
data.columns = data.columns.str.strip()

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
data['target'] = (data['Performance Index'] > 50).astype(int)

data = data.drop_duplicates()
data = data.fillna(data.median())

X = data.drop(['target', 'Performance Index'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- SVM RBF Kernel Results ---")
param_grid_rbf = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 20],
    'gamma': [0.01, 'scale', 'auto'],
    'class_weight': ['balanced', None]
}

grid_search_rbf = GridSearchCV(SVC(), param_grid_rbf, cv=cv_strategy, n_jobs=-1, scoring='accuracy')
grid_search_rbf.fit(X_train_scaled, y_train)

best_model_rbf = grid_search_rbf.best_estimator_
y_pred_rbf = best_model_rbf.predict(X_test_scaled)

results_rbf = {
    "Accuracy": accuracy_score(y_test, y_pred_rbf),
    "F1-Score": f1_score(y_test, y_pred_rbf),
    "Precision": precision_score(y_test, y_pred_rbf),
    "Recall": recall_score(y_test, y_pred_rbf),
    "MCC": matthews_corrcoef(y_test, y_pred_rbf)
}

print(f"Best Params: {grid_search_rbf.best_params_}")
for k, v in results_rbf.items(): print(f"{k}: {v:.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rbf), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix - RBF Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n--- SVM Linear Kernel Results ---")
param_grid_linear = {
    'kernel': ['linear'],
    'C': [0.1, 1, 10, 20],
    'class_weight': ['balanced', None]
}

grid_search_linear = GridSearchCV(SVC(), param_grid_linear, cv=cv_strategy, n_jobs=-1, scoring='accuracy')
grid_search_linear.fit(X_train_scaled, y_train)

best_model_linear = grid_search_linear.best_estimator_
y_pred_linear = best_model_linear.predict(X_test_scaled)

results_linear = {
    "Accuracy": accuracy_score(y_test, y_pred_linear),
    "F1-Score": f1_score(y_test, y_pred_linear),
    "Precision": precision_score(y_test, y_pred_linear),
    "Recall": recall_score(y_test, y_pred_linear),
    "MCC": matthews_corrcoef(y_test, y_pred_linear)
}

print(f"Best Params: {grid_search_linear.best_params_}")
for k, v in results_linear.items(): print(f"{k}: {v:.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_linear), annot=True, fmt='d', cmap='Greens',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix - Linear Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
