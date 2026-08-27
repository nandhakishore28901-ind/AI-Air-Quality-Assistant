import pandas as pd

# ==========================================
# 1. Load cleaned datasets
# ==========================================

historical = pd.read_csv(
    "data/cleaned/historical_aqi_cleaned.csv"
)

pollutant = pd.read_csv(
    "data/cleaned/pollutant_monitoring_cleaned.csv"
)


# ==========================================
# 2. Basic information
# ==========================================

print("\n========== DATASET INFORMATION ==========")

print("\nHistorical AQI:")
print("Rows:", len(historical))
print("Columns:", len(historical.columns))

print("\nPollutant Monitoring:")
print("Rows:", len(pollutant))
print("Columns:", len(pollutant.columns))


# ==========================================
# 3. AQI statistics
# ==========================================

print("\n========== AQI STATISTICS ==========")

print(historical["aqi_value"].describe())


# ==========================================
# 4. Average AQI by state
# ==========================================

print("\n========== TOP 10 STATES BY AVERAGE AQI ==========")

state_aqi = (
    historical
    .groupby("state")["aqi_value"]
    .mean()
    .sort_values(ascending=False)
)

print(state_aqi.head(10))


# ==========================================
# 5. AQI status distribution
# ==========================================

print("\n========== AIR QUALITY STATUS ==========")

status_counts = historical["air_quality_status"].value_counts()

print(status_counts)


# ==========================================
# 6. Most common pollutants
# ==========================================

print("\n========== PROMINENT POLLUTANTS ==========")

pollutant_counts = (
    historical["prominent_pollutants"]
    .value_counts()
)

print(pollutant_counts.head(10))


# ==========================================
# 7. Pollutant monitoring count
# ==========================================

print("\n========== MONITORED POLLUTANTS ==========")

print(
    pollutant["pollutant_id"]
    .value_counts()
)


# ==========================================
# 8. Average pollutant concentration
# ==========================================

print("\n========== AVERAGE POLLUTANT CONCENTRATION ==========")

pollutant_average = (
    pollutant
    .groupby("pollutant_id")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
)

print(pollutant_average)


# ==========================================
# 9. Highest AQI locations
# ==========================================

print("\n========== TOP 10 HIGHEST AQI RECORDS ==========")

highest_aqi = historical[
    ["date", "state", "area", "aqi_value", "air_quality_status"]
].sort_values(
    "aqi_value",
    ascending=False
)

print(highest_aqi.head(10).to_string(index=False))


print("\n========== ANALYSIS COMPLETE ==========")