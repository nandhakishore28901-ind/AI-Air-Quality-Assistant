from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "aqi_prediction_model.pkl"
)

HISTORY_FOLDER = os.path.join(
    BASE_DIR,
    "history"
)

HISTORY_FILE = os.path.join(
    HISTORY_FOLDER,
    "prediction_history.csv"
)

# Create history folder
os.makedirs(HISTORY_FOLDER, exist_ok=True)


# =========================================================
# LOAD ML MODEL
# =========================================================

try:

    model = joblib.load(MODEL_FILE)

    print("==========================================")
    print("AQI MODEL LOADED SUCCESSFULLY!")
    print("Model:", MODEL_FILE)
    print("==========================================")

except Exception as e:

    print("ERROR LOADING MODEL:")
    print(e)

    model = None


# =========================================================
# CREATE HISTORY FILE
# =========================================================

if not os.path.exists(HISTORY_FILE):

    history_columns = [

        "timestamp",
        "state",
        "area",
        "number_of_monitoring_stations",
        "year",
        "month",
        "day",
        "day_of_week",
        "predicted_aqi",
        "category"

    ]

    pd.DataFrame(
        columns=history_columns
    ).to_csv(
        HISTORY_FILE,
        index=False
    )


print("History file:")
print(HISTORY_FILE)


# =========================================================
# AQI INFORMATION
# =========================================================

def get_aqi_information(aqi):

    if aqi <= 50:

        return {

            "category": "Good",

            "health_message":
                "Air quality is satisfactory and poses little or no health risk.",

            "recommendation":
                "Enjoy normal outdoor activities.",

            "risk_level":
                "Low",

            "sustainability_score":
                95,

            "sustainability_actions": [

                "Continue using public transport, walking or cycling when convenient.",

                "Maintain green spaces and trees around the community.",

                "Avoid unnecessary vehicle idling.",

                "Continue responsible waste management.",

                "Support clean-energy and low-emission practices."

            ]

        }

    elif aqi <= 100:

        return {

            "category": "Satisfactory",

            "health_message":
                "Air quality is acceptable, but some sensitive individuals may experience minor effects.",

            "recommendation":
                "Sensitive individuals should consider reducing prolonged outdoor activity.",

            "risk_level":
                "Low to Moderate",

            "sustainability_score":
                80,

            "sustainability_actions": [

                "Prefer public transport or shared transportation.",

                "Reduce unnecessary private vehicle trips.",

                "Avoid burning leaves or other waste.",

                "Maintain trees and vegetation in the surrounding area.",

                "Reduce unnecessary energy consumption."

            ]

        }

    elif aqi <= 200:

        return {

            "category": "Moderate",

            "health_message":
                "Some people may experience health effects, especially sensitive groups.",

            "recommendation":
                "Sensitive individuals should limit prolonged outdoor exertion.",

            "risk_level":
                "Moderate",

            "sustainability_score":
                65,

            "sustainability_actions": [

                "Prefer public transport, carpooling or walking for short trips.",

                "Reduce unnecessary vehicle usage.",

                "Avoid open burning and unnecessary smoke generation.",

                "Reduce activities that create dust.",

                "Increase and protect green spaces.",

                "Use energy-efficient appliances and reduce electricity waste."

            ]

        }

    elif aqi <= 300:

        return {

            "category": "Poor",

            "health_message":
                "Health effects may be experienced by everyone.",

            "recommendation":
                "Reduce prolonged outdoor activity and avoid heavy exertion.",

            "risk_level":
                "High",

            "sustainability_score":
                45,

            "sustainability_actions": [

                "Avoid unnecessary vehicle trips.",

                "Use public transportation whenever possible.",

                "Avoid open waste burning completely.",

                "Reduce dust-generating construction and outdoor activities.",

                "Increase vegetation and dust-control measures.",

                "Reduce industrial and household emissions where possible."

            ]

        }

    elif aqi <= 400:

        return {

            "category": "Very Poor",

            "health_message":
                "Risk of health effects is increased for everyone.",

            "recommendation":
                "Avoid prolonged outdoor activity, especially strenuous exercise.",

            "risk_level":
                "Very High",

            "sustainability_score":
                25,

            "sustainability_actions": [

                "Minimize private vehicle usage.",

                "Use public transportation or remote alternatives.",

                "Stop open burning and unnecessary combustion activities.",

                "Avoid dust-producing activities.",

                "Implement strong emission-reduction measures.",

                "Increase monitoring of major pollution sources.",

                "Protect and expand vegetation wherever possible."

            ]

        }

    else:

        return {

            "category": "Severe",

            "health_message":
                "Health alert: serious health effects may occur for everyone.",

            "recommendation":
                "Avoid outdoor exposure as much as possible.",

            "risk_level":
                "Critical",

            "sustainability_score":
                10,

            "sustainability_actions": [

                "Avoid unnecessary outdoor travel.",

                "Minimize all avoidable vehicle emissions.",

                "Stop open burning and combustion activities.",

                "Implement emergency pollution-control measures.",

                "Restrict dust-generating activities.",

                "Increase air-quality monitoring and public alerts.",

                "Promote immediate emission-reduction actions."

            ]

        }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    return render_template(
        "dashboard.html"
    )


# =========================================================
# API STATUS
# =========================================================

@app.route("/api", methods=["GET"])
def api_status():

    return jsonify({

        "message":
            "AI Air Quality Prediction API is running",

        "status":
            "success"

    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "AQI model could not be loaded."

            }), 500


        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No input data received."

            }), 400


        required_fields = [

            "state",
            "area",
            "number_of_monitoring_stations",
            "year",
            "month",
            "day",
            "day_of_week"

        ]


        # -------------------------------------------------
        # VALIDATE INPUT
        # -------------------------------------------------

        for field in required_fields:

            if field not in data:

                return jsonify({

                    "success": False,

                    "error":
                        f"Missing field: {field}"

                }), 400


        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "state": [
                data["state"]
            ],

            "area": [
                data["area"]
            ],

            "number_of_monitoring_stations": [
                data["number_of_monitoring_stations"]
            ],

            "year": [
                data["year"]
            ],

            "month": [
                data["month"]
            ],

            "day": [
                data["day"]
            ],

            "day_of_week": [
                data["day_of_week"]
            ]

        })


        # -------------------------------------------------
        # ML PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )


        predicted_aqi = round(
            float(prediction[0]),
            2
        )


        # Prevent unrealistic negative AQI
        predicted_aqi = max(
            0,
            predicted_aqi
        )


        # -------------------------------------------------
        # AQI INFORMATION
        # -------------------------------------------------

        aqi_info = get_aqi_information(
            predicted_aqi
        )


        # -------------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------------

        history_record = pd.DataFrame([{

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "state":
                data["state"],

            "area":
                data["area"],

            "number_of_monitoring_stations":
                data["number_of_monitoring_stations"],

            "year":
                data["year"],

            "month":
                data["month"],

            "day":
                data["day"],

            "day_of_week":
                data["day_of_week"],

            "predicted_aqi":
                predicted_aqi,

            "category":
                aqi_info["category"]

        }])


        history_record.to_csv(

            HISTORY_FILE,

            mode="a",

            header=False,

            index=False

        )


        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return jsonify({

            "success":
                True,

            "predicted_aqi":
                predicted_aqi,

            "category":
                aqi_info["category"],

            "health_message":
                aqi_info["health_message"],

            "recommendation":
                aqi_info["recommendation"],

            "risk_level":
                aqi_info["risk_level"],

            "sustainability_score":
                aqi_info["sustainability_score"],

            "sustainability_actions":
                aqi_info["sustainability_actions"]

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# =========================================================
# PREDICTION HISTORY API
# =========================================================

@app.route("/history", methods=["GET"])
def history():

    try:

        history_data = pd.read_csv(
            HISTORY_FILE
        )


        if history_data.empty:

            return jsonify({

                "success":
                    True,

                "history":
                    []

            })


        # Latest first
        history_data = history_data.iloc[::-1]


        records = history_data.to_dict(
            orient="records"
        )


        return jsonify({

            "success":
                True,

            "history":
                records

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )