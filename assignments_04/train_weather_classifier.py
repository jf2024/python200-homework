import requests
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn
import json
import sys
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

# Step 1
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.579449,
    "longitude": -118.109291,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(df.head())
print(df.describe())
print(df.info())

# Step 2
df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 30) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
)

print(df["good_for_running"].value_counts())
fraction_good = df["good_for_running"].mean()
print("Fraction of days good for running:", fraction_good)
print("Percentage of days good for running:", fraction_good * 100)

# I decided to keep the defauly thresholds as it made sense for my location, which is 
    # palmdale, ca. A desert, windy place. I think this was the right decision since the 
    # percentage of good days were running was about 51% so a nice even split 

# Step 3
X = df[[
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] #will use what we did for the warmup
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print(f"Best C:      {grid_search.best_params_['clf__C']}")  #100
print(f"Best CV AUC: {grid_search.best_score_:.3f}")  #0.871
best_pipe = grid_search.best_estimator_
y_pred  = best_pipe.predict(X_test) 
y_probs = best_pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
test_auc = roc_auc_score(y_test, y_probs)
print(f"Test AUC: {test_auc:.3f}") #0.888

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve — Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.show()

# Step 4

# The test AUC of 0.888 shows that the logistic regression model does a good job
    # distinguishing between days that are good for running and days that aren't .
    # The AUC is high because the target label that we created was derived from th e 
    # features used to train the model so the strong performance shouldn't be too suprising. 
# The classification report shows that false negatives are slightly more common
# than false positives: recall for True is 0.76, while recall for False is 0.86.
# For the app, it would under-recommend running than over-recommend it. 
    # I think i would rather it do the slightly under-recommend running because
    # incorrectly telling someone that weather is good could lead them to run in potential
    # harmful weather 
# I would keep the default 0.5 threshold because the precision and
    # recall are both reasonably balanced.

# Step 5
best_pipe = grid_search.best_estimator_
joblib.dump(best_pipe, "models/weather_classifier.pkl")
print("weather classifer model saved.")

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

metadata = {
    "python_version":  sys.version,
    "sklearn_version": sklearn.__version__,
    "features":        FEATURES,
    "label":           "good_for_running",
    "best_params":     grid_search.best_params_,
    "test_auc":        round(test_auc, 4),
    "trained_on":      "2023 Open-Meteo, Palmdale CA (lat 34.579449, lon -118.109291)",

    "label_definition": {
        "description": "Good conditions for running based on temperature, precipitation, and wind.",
        "rules": {
            "temperature_2m_max": ">= 7°C and <= 30°C",
            "temperature_2m_min": ">= 0°C",
            "precipitation_sum": "< 3.0 mm",
            "wind_speed_10m_max": "< 30 km/h",
        },
        "logic": "All conditions must be true",
        "positive_label": 1,
        "negative_label": 0,
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("metadata saved to models")
