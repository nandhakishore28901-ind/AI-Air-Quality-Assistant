import pandas as pd
import joblib


# ==========================================
# 1. Load trained model
# ==========================================

model = joblib.load("ai/aqi_prediction_model.pkl")

print("AQI model loaded successfully!")


# ==========================================
# 2. Create new input
# ==========================================

new_data = pd.DataFrame({
    "state": ["Maharashtra"],
    "area": ["Mumbai"],
    "number_of_monitoring_stations": [10],
    "year": [2026],
    "month": [8],
    "day": [25],
    "day_of_week": [1]
})


# ==========================================
# 3. Predict AQI
# ==========================================

prediction = model.predict(new_data)


# ==========================================
# 4. Display result
# ==========================================

print("\n========== AQI PREDICTION ==========")
print("Predicted AQI:", round(prediction[0], 2))