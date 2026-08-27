import pandas as pd
import os

# -----------------------------
# 1. Load the datasets
# -----------------------------

historical_path = "data/historical_aqi.csv"
pollutant_path = "data/pollutant_monitoring.csv"

historical = pd.read_csv(historical_path)
pollutant = pd.read_csv(pollutant_path)

print("Original Historical AQI shape:", historical.shape)
print("Original Pollutant shape:", pollutant.shape)


# -----------------------------
# 2. Clean Historical AQI data
# -----------------------------

# Remove completely empty column
if "note" in historical.columns:
    historical = historical.drop(columns=["note"])

# Convert date column to datetime
historical["date"] = pd.to_datetime(
    historical["date"],
    dayfirst=True,
    errors="coerce"
)


# -----------------------------
# 3. Clean Pollutant data
# -----------------------------

# Convert last_update to datetime
pollutant["last_update"] = pd.to_datetime(
    pollutant["last_update"],
    errors="coerce"
)

# Remove rows where all pollutant measurements are missing
pollutant = pollutant.dropna(
    subset=["pollutant_min", "pollutant_max", "pollutant_avg"],
    how="all"
)


# -----------------------------
# 4. Create cleaned-data folder
# -----------------------------

os.makedirs("data/cleaned", exist_ok=True)


# -----------------------------
# 5. Save cleaned datasets
# -----------------------------

historical.to_csv(
    "data/cleaned/historical_aqi_cleaned.csv",
    index=False
)

pollutant.to_csv(
    "data/cleaned/pollutant_monitoring_cleaned.csv",
    index=False
)


# -----------------------------
# 6. Display results
# -----------------------------

print("\nCleaning completed!")

print("Cleaned Historical AQI shape:", historical.shape)
print("Cleaned Pollutant shape:", pollutant.shape)

print("\nCleaned files created:")
print("data/cleaned/historical_aqi_cleaned.csv")
print("data/cleaned/pollutant_monitoring_cleaned.csv")
