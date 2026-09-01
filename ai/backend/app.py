import os
import json
from datetime import datetime

import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

# Model locations
MODEL_PATH_BACKEND = os.path.join(
    BASE_DIR,
    "aqi_prediction_model.pkl"
)

MODEL_PATH_AI = os.path.join(
    os.path.dirname(BASE_DIR),
    "aqi_prediction_model.pkl"
)

HISTORY_FILE = os.path.join(
    HISTORY_DIR,
    "prediction_history.json"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)


# ============================================================
# LOAD AQI MODEL
# ============================================================

model = None
MODEL_USED = None

MODEL_PATHS = [
    MODEL_PATH_BACKEND,
    MODEL_PATH_AI
]

print("=" * 60)
print("LOADING AQI MODEL")
print("=" * 60)

for model_path in MODEL_PATHS:

    if os.path.exists(model_path):

        try:

            print("Checking model:")
            print(model_path)

            model = joblib.load(model_path)

            MODEL_USED = model_path

            print("AQI MODEL LOADED SUCCESSFULLY")
            print("=" * 60)

            break

        except Exception as error:

            print("ERROR LOADING MODEL:")
            print(error)
            print("=" * 60)


if model is None:

    print("WARNING: AQI MODEL COULD NOT BE LOADED")

    for path in MODEL_PATHS:
        print("Checked:", path)

    print("=" * 60)


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(aqi):

    if aqi <= 50:
        return "Low"

    elif aqi <= 100:
        return "Low to Moderate"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "High"

    elif aqi <= 400:
        return "Very High"

    else:
        return "Critical"


# ============================================================
# HEALTH INFORMATION
# ============================================================

def get_health_information(aqi):

    if aqi <= 50:

        return (
            "Air quality is good. "
            "Outdoor activities are generally safe."
        )

    elif aqi <= 100:

        return (
            "Air quality is satisfactory. "
            "Sensitive individuals should monitor air quality."
        )

    elif aqi <= 200:

        return (
            "Air quality may affect sensitive individuals. "
            "Reduce prolonged outdoor exposure if necessary."
        )

    elif aqi <= 300:

        return (
            "Poor air quality may affect everyone. "
            "Avoid prolonged or strenuous outdoor activity."
        )

    elif aqi <= 400:

        return (
            "Very poor air quality. "
            "Avoid outdoor exposure as much as possible."
        )

    else:

        return (
            "Severe air pollution. "
            "Avoid outdoor exposure and follow local health guidance."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendation(aqi):

    if aqi <= 50:

        return (
            "Enjoy outdoor activities and continue maintaining "
            "good environmental practices."
        )

    elif aqi <= 100:

        return (
            "Outdoor activities are generally acceptable. "
            "Sensitive people should monitor air quality."
        )

    elif aqi <= 200:

        return (
            "Reduce prolonged outdoor activity and avoid "
            "unnecessary exposure to polluted areas."
        )

    elif aqi <= 300:

        return (
            "Avoid strenuous outdoor activities. "
            "Keep indoor areas well protected from polluted air."
        )

    elif aqi <= 400:

        return (
            "Avoid outdoor exposure as much as possible "
            "and reduce unnecessary travel."
        )

    else:

        return (
            "Stay indoors where possible and avoid outdoor exposure. "
            "Follow local health recommendations."
        )


# ============================================================
# SUSTAINABILITY ACTIONS
# ============================================================

def get_sustainability_actions(aqi):

    if aqi <= 50:

        return [
            "Walk or cycle for short-distance travel.",
            "Use public transportation when practical.",
            "Maintain trees and green spaces.",
            "Continue energy-efficient practices."
        ]

    elif aqi <= 100:

        return [
            "Prefer public transportation or carpooling.",
            "Avoid unnecessary vehicle idling.",
            "Increase greenery around buildings.",
            "Reduce unnecessary energy consumption."
        ]

    elif aqi <= 200:

        return [
            "Reduce unnecessary vehicle usage.",
            "Prefer public transportation and carpooling.",
            "Avoid open burning of waste.",
            "Increase trees and vegetation around the campus."
        ]

    elif aqi <= 300:

        return [
            "Minimize vehicle trips.",
            "Promote carpooling and public transportation.",
            "Avoid dust-generating activities when possible.",
            "Do not burn leaves or waste."
        ]

    elif aqi <= 400:

        return [
            "Significantly reduce vehicle usage.",
            "Avoid unnecessary outdoor activities.",
            "Control dust-producing activities.",
            "Increase green buffers around polluted areas."
        ]

    else:

        return [
            "Minimize unnecessary outdoor exposure.",
            "Reduce vehicle activity.",
            "Stop open burning and avoid pollution sources.",
            "Implement immediate pollution-control measures."
        ]


# ============================================================
# SUSTAINABILITY SCORE
# ============================================================

def get_sustainability_score(aqi):

    if aqi <= 50:
        return 1

    elif aqi <= 100:
        return 3

    elif aqi <= 200:
        return 5

    elif aqi <= 300:
        return 7

    elif aqi <= 400:
        return 9

    else:
        return 10


# ============================================================
# SAVE HISTORY
# ============================================================

def save_prediction_history(record):

    try:

        history = []

        if os.path.exists(HISTORY_FILE):

            try:

                with open(
                    HISTORY_FILE,
                    "r",
                    encoding="utf-8"
                ) as file:

                    history = json.load(file)

            except Exception:

                history = []

        if not isinstance(history, list):
            history = []

        history.insert(0, record)

        # Keep latest 100 predictions
        history = history[:100]

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4
            )

        return True

    except Exception as error:

        print("HISTORY SAVE ERROR:")
        print(error)

        return False


# ============================================================
# LOAD HISTORY
# ============================================================

def load_prediction_history():

    try:

        if not os.path.exists(HISTORY_FILE):
            return []

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

        if not isinstance(history, list):
            return []

        return history

    except Exception as error:

        print("HISTORY LOAD ERROR:")
        print(error)

        return []


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    return render_template("dashboard.html")


# ============================================================
# API STATUS
# ============================================================

@app.route("/api", methods=["GET"])
def api_status():

    return jsonify({
        "success": True,
        "status": "running",
        "message": "AI Air Quality Prediction API is running",
        "model_loaded": model is not None
    })


# ============================================================
# HISTORY API
# ============================================================

@app.route("/history", methods=["GET"])
@app.route("/api/history", methods=["GET"])
def history():

    prediction_history = load_prediction_history()

    return jsonify({
        "success": True,
        "history": prediction_history
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        print("=" * 60)
        print("PREDICTION REQUEST RECEIVED")
        print("=" * 60)

        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            print("MODEL IS NOT LOADED")

            return jsonify({
                "success": False,
                "error": "AQI model could not be loaded.",
                "prediction_error": True
            }), 500


        # ====================================================
        # GET REQUEST DATA
        # ====================================================

        data = request.get_json(silent=True)

        print("Received data:")
        print(data)

        if not data:

            return jsonify({
                "success": False,
                "error": "No input data received.",
                "prediction_error": True
            }), 400


        # ====================================================
        # GET STATE
        # ====================================================

        state = str(
            data.get("state", "")
        ).strip()


        # ====================================================
        # GET AREA
        # ====================================================

        area = str(
            data.get("area", "")
        ).strip()


        # ====================================================
        # GET MONITORING STATIONS
        # ====================================================

        number_of_monitoring_stations = data.get(
            "number_of_monitoring_stations",
            1
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if not state:

            return jsonify({
                "success": False,
                "error": "State is required."
            }), 400


        if not area:

            return jsonify({
                "success": False,
                "error": "Area is required."
            }), 400


        # ====================================================
        # CONVERT STATIONS
        # ====================================================

        try:

            number_of_monitoring_stations = int(
                number_of_monitoring_stations
            )

        except (
            ValueError,
            TypeError
        ):

            number_of_monitoring_stations = 1


        if number_of_monitoring_stations < 1:

            number_of_monitoring_stations = 1


        # ====================================================
        # CURRENT DATE FEATURES
        # ====================================================

        now = datetime.now()

        year = now.year
        month = now.month
        day = now.day
        day_of_week = now.weekday()


        # ====================================================
        # MODEL INPUT
        # ====================================================

        input_data = pd.DataFrame([{

            "state": state,

            "area": area,

            "number_of_monitoring_stations":
                number_of_monitoring_stations,

            "year": year,

            "month": month,

            "day": day,

            "day_of_week": day_of_week

        }])


        print("=" * 60)
        print("INPUT SENT TO MODEL")
        print("=" * 60)

        print(input_data)

        print("=" * 60)


        # ====================================================
        # ML PREDICTION
        # ====================================================

        prediction = model.predict(
            input_data
        )[0]


        predicted_aqi = float(
            prediction
        )


        # ====================================================
        # PREVENT INVALID NEGATIVE AQI
        # ====================================================

        predicted_aqi = max(
            0,
            predicted_aqi
        )


        predicted_aqi = round(
            predicted_aqi,
            2
        )


        # ====================================================
        # GENERATE AQI INFORMATION
        # ====================================================

        category = get_aqi_category(
            predicted_aqi
        )

        risk_level = get_risk_level(
            predicted_aqi
        )

        health_information = get_health_information(
            predicted_aqi
        )

        recommendation = get_recommendation(
            predicted_aqi
        )

        sustainability_actions = (
            get_sustainability_actions(
                predicted_aqi
            )
        )

        sustainability_score = (
            get_sustainability_score(
                predicted_aqi
            )
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp = now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        # ====================================================
        # HISTORY RECORD
        # ====================================================

        history_record = {

            "timestamp":
                timestamp,

            "state":
                state,

            "area":
                area,

            "number_of_monitoring_stations":
                number_of_monitoring_stations,

            "year":
                year,

            "month":
                month,

            "day":
                day,

            "day_of_week":
                day_of_week,

            "predicted_aqi":
                predicted_aqi,

            "category":
                category,

            "risk_level":
                risk_level,

            "health_information":
                health_information,

            "recommendation":
                recommendation,

            "sustainability_score":
                sustainability_score,

            "sustainability_actions":
                sustainability_actions

        }


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        save_prediction_history(
            history_record
        )


        # ====================================================
        # FINAL API RESPONSE
        #
        # Multiple aliases are intentionally included so
        # the existing frontend can use either naming style.
        # ====================================================

        result = {

            "success":
                True,

            # AQI
            "predicted_aqi":
                predicted_aqi,

            "aqi":
                predicted_aqi,

            "prediction":
                predicted_aqi,


            # CATEGORY
            "category":
                category,

            "aqi_category":
                category,


            # RISK
            "risk_level":
                risk_level,

            "risk":
                risk_level,


            # HEALTH
            "health_information":
                health_information,

            "health_info":
                health_information,

            "health_message":
                health_information,

            "health":
                health_information,


            # RECOMMENDATION
            "recommendation":
                recommendation,

            "recommendations":
                recommendation,


            # SUSTAINABILITY
            "sustainability_actions":
                sustainability_actions,

            "sustainability":
                sustainability_actions,

            "actions":
                sustainability_actions,


            # SCORE
            "sustainability_score":
                sustainability_score,


            # INPUT INFORMATION
            "state":
                state,

            "area":
                area,

            "number_of_monitoring_stations":
                number_of_monitoring_stations,


            # DATE
            "year":
                year,

            "month":
                month,

            "day":
                day,

            "day_of_week":
                day_of_week,


            # TIME
            "timestamp":
                timestamp

        }


        print("=" * 60)
        print("PREDICTION SUCCESSFUL")
        print("=" * 60)

        print("AQI:", predicted_aqi)
        print("CATEGORY:", category)
        print("RISK:", risk_level)
        print("HEALTH:", health_information)
        print("RECOMMENDATION:", recommendation)
        print("SUSTAINABILITY SCORE:", sustainability_score)
        print("SUSTAINABILITY ACTIONS:")
        print(sustainability_actions)

        print("=" * 60)


        return jsonify(
            result
        )


    except Exception as error:

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            type(error).__name__
        )

        print(
            str(error)
        )

        print("=" * 60)


        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "prediction_error":
                True

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status":
            "running",

        "model_loaded":
            model is not None,

        "model_path":
            MODEL_USED,

        "templates_path":
            TEMPLATES_DIR,

        "templates_exist":
            os.path.exists(
                TEMPLATES_DIR
            ),

        "index_exists":
            os.path.exists(
                os.path.join(
                    TEMPLATES_DIR,
                    "index.html"
                )
            ),

        "dashboard_exists":
            os.path.exists(
                os.path.join(
                    TEMPLATES_DIR,
                    "dashboard.html"
                )
            ),

        "static_path":
            STATIC_DIR,

        "static_exists":
            os.path.exists(
                STATIC_DIR
            ),

        "history_file":
            HISTORY_FILE,

        "history_exists":
            os.path.exists(
                HISTORY_FILE
            )

    })


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI AIR QUALITY ASSISTANT")
    print("=" * 60)

    print("Backend folder:")
    print(BASE_DIR)

    print()

    print("Templates folder:")
    print(TEMPLATES_DIR)

    print()

    print("Static folder:")
    print(STATIC_DIR)

    print()

    print("History folder:")
    print(HISTORY_DIR)

    print()

    print("Model:")
    print(MODEL_USED)

    print()

    print("Model loaded:")
    print(model is not None)

    print("=" * 60)

    # Render provides PORT through environment variable.
    # Local testing defaults to 5000.
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )