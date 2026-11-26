import pandas as pd
from sklearn.preprocessing import StandardScaler

folder = "data/processed_data"

X_train = pd.read_csv(f"{folder}/X_train.csv")
X_test = pd.read_csv(f"{folder}/X_test.csv")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train_scaled.to_csv(f"{folder}/X_train_scaled.csv", index=False)
X_test_scaled.to_csv(f"{folder}/X_test_scaled.csv", index=False)

print("Normalized data saved")
