import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

folder = "data/processed_data"

X_test = pd.read_csv(f"{folder}/X_test_scaled.csv")
y_test = pd.read_csv(f"{folder}/y_test.csv").values.ravel()

model = joblib.load("models/trained_model.pkl")
y_pred = model.predict(X_test)

predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
predictions.to_csv("data/predictions.csv", index=False)

mse = mean_squared_error(y_test, y_pred)
scores = {
    "mse": mse,
    "rmse": mse ** 0.5,
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(scores)
