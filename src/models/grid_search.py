import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

folder = "data/processed_data"

X_train = pd.read_csv(f"{folder}/X_train_scaled.csv")
y_train = pd.read_csv(f"{folder}/y_train.csv").values.ravel()

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_params_, "models/best_params.pkl")

print(f"Best parameters: {grid_search.best_params_}")
