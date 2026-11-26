import pandas as pd
from sklearn.model_selection import train_test_split

raw_path = "data/raw_data/raw.csv"
output_folder = "data/processed_data"

df = pd.read_csv(raw_path)
df = df.select_dtypes(include="number")

X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv(f"{output_folder}/X_train.csv", index=False)
X_test.to_csv(f"{output_folder}/X_test.csv", index=False)
y_train.to_csv(f"{output_folder}/y_train.csv", index=False)
y_test.to_csv(f"{output_folder}/y_test.csv", index=False)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
