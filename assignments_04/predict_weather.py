import joblib
import pandas as pd
import json


# Step 1
clf = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"City: {metadata['trained_on']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']}")

# Step 2
new_days = pd.DataFrame({
    "temperature_2m_max": [18.0, 35.0, 22.0, 16.0, 26.0],
    "temperature_2m_min": [10.0, 20.0, 10.0, 9.0, 0.0],
    "precipitation_sum":  [0.0,   0.0,  0.5,  8.0, 2.9],
    "wind_speed_10m_max": [18.0,  35.0, 18.0, 15.0, 29.0],
})

new_days = new_days[metadata["features"]]

preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"

    print(f"Temperature max: {new_days.iloc[i]['temperature_2m_max']} °C")
    print(f"Temperature min: {new_days.iloc[i]['temperature_2m_min']} °C")
    print(f"Precipitation:   {new_days.iloc[i]['precipitation_sum']} mm")
    print(f"Wind speed max:  {new_days.iloc[i]['wind_speed_10m_max']} km/h")
    print(f"Prediction:      {label}")
    print(f"Confidence:      {prob:.2f}")

# Step 3
"""

1. The borderline case was the last one since all 4 of the conditions were
close to the thresholds, they were: max temp (26 C), mi temp (0 C), precipitation (2.9mm), 
and max wind (29 km/h). The model prediced skip with a 0.01 confidence which means its confidence for
"good weather for running" is at 1% and 99% for skip. So the model is very confident for this Day. However,
if it was 0.52, it means its more 50/50 and the model is more uncertain (basically a coin toss), so I would
look into more weather information/data before making a decision for that specific day.

2. If we ran the predict_weather file before the train file, it would give us that our weather classifer model
doesn't exist which is true since we need to run the train one first. There would be no model to load and do predictions.
I would probably say something "run the train weather classifer file before the predict_weather" and that should help.

3. If we were to do this daily, we would need to get the weather data for tomorrow's forecast and replace our hypothetical 
dataframe grabbing the same 4 features as before. Then it would just go through the pipeline and we can set the script
to be schedule at a specific time to be run everyday and say if that day is good for a run or not. 

"""