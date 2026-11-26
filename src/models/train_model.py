import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

folder = "data/processed_data"

X_train = pd.read_csv(f"{folder}/X_train_scaled.csv")
y_train = pd.read_csv(f"{folder}/y_train.csv").values.ravel()

best_params = joblib.load("models/best_params.pkl")

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/trained_model.pkl")

print("Model trained and saved")
