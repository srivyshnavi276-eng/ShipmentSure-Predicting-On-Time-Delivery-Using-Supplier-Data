import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("model", exist_ok=True)

df = pd.read_csv("Train.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

df.columns = df.columns.str.strip()

target_column = "Reached.on.Time_Y.N"

if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=[target_column])
y = df[target_column]

for col in X.columns:
    if X[col].dtype == "object":
        mode_vals = X[col].mode()
        X[col] = X[col].fillna(mode_vals[0] if not mode_vals.empty else "Unknown")
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

X = pd.get_dummies(X, drop_first=True)
model_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/shipment_model.pkl")
joblib.dump(model_columns, "model/model_columns.pkl")

print("\nSaved files:")
print("model/shipment_model.pkl")
print("model/model_columns.pkl")