import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load cleaned dataset
# ==========================================

df = pd.read_csv("data/cleaned/historical_aqi_cleaned.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. Convert date
# ==========================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")


# ==========================================
# 3. Feature Engineering
# ==========================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek


# ==========================================
# 4. Remove rows with invalid target/date
# ==========================================

df = df.dropna(subset=["aqi_value", "date"])


# ==========================================
# 5. Define target and features
# ==========================================

target = "aqi_value"

features = [
    "state",
    "area",
    "number_of_monitoring_stations",
    "year",
    "month",
    "day",
    "day_of_week"
]

X = df[features]
y = df[target]


# ==========================================
# 6. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. Identify feature types
# ==========================================

categorical_features = [
    "state",
    "area"
]

numeric_features = [
    "number_of_monitoring_stations",
    "year",
    "month",
    "day",
    "day_of_week"
]


# ==========================================
# 8. Preprocessing
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 9. Optimized Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 10. Complete ML Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 11. Train Model
# ==========================================

print("\nTraining model...")
print("Please wait...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# ==========================================
# 12. Predictions
# ==========================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)


# ==========================================
# 13. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# ==========================================
# 14. Compare Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual_AQI": y_test.values,
    "Predicted_AQI": np.round(y_pred, 2)
})


print("\n========== ACTUAL VS PREDICTED ==========")

print(comparison.head(10))


# ==========================================
# 15. Save Trained Model
# ==========================================

joblib.dump(
    pipeline,
    "ai/aqi_prediction_model.pkl"
)

print("\nModel saved successfully!")

print("\n========== MODEL TRAINING COMPLETE ==========")