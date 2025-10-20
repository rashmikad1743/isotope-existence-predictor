import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# === Load dataset ===
file_path = "final_corrected_isotope_data.csv"
df = pd.read_csv(file_path)

print("[INFO] Dataset loaded successfully. Shape:", df.shape)

# === Clean dataset ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.applymap(lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > 1e10 else x)

# === Define target ===
target_col = 'Predicted Existence Probability (%)'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

y = df[target_col]
X = df.drop(columns=[target_col])

# === Drop irrelevant or high-cardinality columns ===
drop_cols = ['Element']
X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True, errors='ignore')

# === Identify feature types ===
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"[INFO] Numeric features: {len(numeric_features)}")
print(f"[INFO] Categorical features: {len(categorical_features)}")

# === Preprocessing ===
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# === Model ===
model = GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=5)

# === Pipeline ===
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# === K-Fold Cross Validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(reg, X, y, cv=cv, scoring='r2', n_jobs=-1)
print(f"[INFO] Cross-validation R² scores: {scores}")
print(f"[INFO] Mean R²: {np.mean(scores):.4f}")

# === Train final model ===
reg.fit(X, y)
print("[INFO] Final regression model trained successfully.")

# === Evaluate on training data ===
y_pred = reg.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"[INFO] Training MAE: {mae:.4f}")
print(f"[INFO] Training R²: {r2:.4f}")

# === Save model and metadata for Streamlit app ===
import os
import pickle

meta = {
    "model": reg,
    "feature_columns": X.columns.tolist(),
    "target_col": target_col
}

os.makedirs("artifact", exist_ok=True)
with open("artifact/model.pkl", "wb") as f:
    pickle.dump(meta, f)
print("[INFO] Model and metadata saved as artifact/model.pkl")

# === Example prediction ===
sample = X.sample(1, random_state=42)
pred = reg.predict(sample)
print("\n[INFO] Example sample prediction:")
print(sample)
print(f"→ Predicted Existence Probability: {pred[0]:.2f}%")