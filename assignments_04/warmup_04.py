import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=2, #original code had 4, had to change it to 2 
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ROC and AUC --- 

## Q1
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_y_probs = lr_model.predict_proba(X_test)[:, 1] 
lr_auc = roc_auc_score(y_test, lr_y_probs)
print("lr predict proba", lr_y_probs)
print("lr roc_auc", lr_auc) #0.9254

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)
print('knn predict proba', knn_probs)
print('knn roc_auc', knn_auc) #0.917

# logistic regression just barely higher then the knn (0.9254 vs 0.917)
# A higher AUC means it can separate the two classes across all of the thresholds regardless   
    # of the cutoff we give it 

## Q2
fpr, tpr, thresholds = roc_curve(y_test, lr_y_probs)
knn_f, knn_t, knn_thres = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC={lr_auc:.2f})")
RocCurveDisplay(fpr=knn_f, tpr=knn_t).plot(ax=ax, name=f"KNN k=5 (AUC={knn_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC Comparison KNN vs Logistic Regression")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.show()

# At TPR = 0.80, KNN has a slightly lower FPR than Logistic Regression.
    # if we need to detect 80% of the positive cases, KNN would
    # produce fewer false alarms at that operating point.

## Q3
f1_scores = []

for t in thresholds:
    y_pred = (lr_y_probs >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

print("Optimal threshold:", best_threshold)
print("TPR at optimal threshold:", best_tpr)
print("FPR at optimal threshold:", best_fpr)
print("F1 at optimal threshold:", best_f1)

threshold_df = pd.DataFrame({
    "threshold": thresholds,
    "fpr":       fpr,
    "tpr":       tpr,
}).round(3)

target_tpr = 0.50
idx = np.argmin(np.abs(threshold_df["tpr"] - target_tpr))
print(threshold_df.iloc[idx])

# The optimal threshold is 0.32, which is lower than the
    # default threshold of 0.5. 
# Lowering the threshold makes the model predict more positives and increases false
    # positives. In a real world scenario, we might lower it if we believe that 
    # missing a positive test case is more costly/severe then a false alarm

# --- GridSearch --- 

# Q1
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print(f"Best C:      {grid_search.best_params_['clf__C']}") #10
print(f"Best CV AUC: {grid_search.best_score_:.3f}") #0.937
best_pipe = grid_search.best_estimator_
y_pred  = best_pipe.predict(X_test)         # no manual scaling needed
y_probs = best_pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}") #0.925

# My best C was 10, best CV auc was 0.937 and the Test AUC was 0.925
# I think my original guess would have been 1 or 10 and in terms of change between 100 and 1, the 
    # test AUC is a bit smaller but not by much.

## Q2
pipe2 = Pipeline([
    ("clf", DecisionTreeClassifier(random_state=42)),
])

param_grid2 = {
    "clf__max_depth": [2, 3, 5, 8, None]
}

grid_search2 = GridSearchCV(
    estimator=pipe2,
    param_grid=param_grid2,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid_search2.fit(X_train, y_train)

print(f"Best max_depth: {grid_search2.best_params_['clf__max_depth']}") #5
print(f"Best CV AUC:    {grid_search2.best_score_:.3f}")    #0.930
best_tree = grid_search2.best_estimator_
tree_probs = best_tree.predict_proba(X_test)[:, 1]
tree_auc = roc_auc_score(y_test, tree_probs)
print(f"Test AUC:       {tree_auc:.3f}")    #0.889

# Logistic Regression better with test auc at 0.925 to the decision tree at 0.889
# I would bring the logistic regressoin for further development and finetuning purposes. 
# We should also consider other factors like interpretability and speed (decision trees can be slower then logistic regressoin)
    # especially if we check lots of different depts 

## Q3
results = pd.DataFrame(grid_search2.cv_results_)

print(
    results[["param_clf__max_depth", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)

# max_depth=5 and max_depth=3 have relatively similar mean CV AUC scores (0.93 and 0.92).
    # max_depth=5 has a lower standard deviation (0.0090 vs. 0.013)
    # meaning its performance is slightly more consistent across the CV folds.
# I think we should go with max depth 5 as it has a higher average and more stability

# --- Joblib --- 

## Q1
best_pipe = grid_search.best_estimator_
joblib.dump(best_pipe, "models/warmup_model.pkl")
print("Model saved, logistic regression scaled.")

loaded_clf = joblib.load("models/warmup_model.pkl")
original_preds = best_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# If we saved only the logistic regression model without the scaler,
    # .predict(X_test) would receive unscaled data, which is bad because the 
    # was trained on scaled data. This could lead to incorrect predictions
    # since the lr model expects the same scaled feature values it saw during training.

## Q2

# --- Simulated prediction script ---

import numpy as np

loaded_model = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

predicted_classes = loaded_model.predict(new_samples)
predicted_probabilities = loaded_model.predict_proba(new_samples)

for i in range(len(new_samples)):
    predicted_class = predicted_classes[i]
    probability = predicted_probabilities[i, predicted_class]

    print(
        f"Row {i + 1}: "
        f"predicted class = {predicted_class}, "
        f"probability = {probability:.4f}"
    )

# The all-zeros row is around the average after scaling, 
    # so the model will likely be less certain about its prediction and be closer to the decision boundary, in this
    # case it prediced the class to be 1 
