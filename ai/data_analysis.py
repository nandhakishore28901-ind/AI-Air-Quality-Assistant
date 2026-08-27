import pandas as pd

# Load the datasets
historical_aqi = pd.read_csv("data/historical_aqi.csv")
pollutant_data = pd.read_csv("data/pollutant_monitoring.csv")

# Display basic information
print("HISTORICAL AQI DATA")
print("-------------------")
print("Rows:", len(historical_aqi))
print("Columns:", len(historical_aqi.columns))
print()

print("Columns:")
print(historical_aqi.columns.tolist())
print()

print("POLLUTANT MONITORING DATA")
print("-------------------------")
print("Rows:", len(pollutant_data))
print("Columns:", len(pollutant_data.columns))
print()

print("Columns:")
print(pollutant_data.columns.tolist())