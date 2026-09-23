from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Load trained model pipeline
model = joblib.load("trained_model.pkl")


# Create FastAPI application
app = FastAPI(title="AQI Prediction API")


# Input format for /predict
class AQIInput(BaseModel):
    pollutant_min: float
    pollutant_max: float
    pollutant_avg: float
    temperature_c: float
    humidity_percent: float
    wind_speed_kmh: float
    Digital_Elevation_Model: float
    pollutant_id: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AQI Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: AQIInput):

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "pollutant_min": data.pollutant_min,
        "pollutant_max": data.pollutant_max,
        "pollutant_avg": data.pollutant_avg,
        "temperature_c": data.temperature_c,
        "humidity_percent": data.humidity_percent,
        "wind_speed_kmh": data.wind_speed_kmh,
        "Digital Elevation Model": data.Digital_Elevation_Model,
        "pollutant_id": data.pollutant_id
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probability
    probability = model.predict_proba(input_data)[0]

    # Convert prediction into category
    if prediction == 0:
        category = "Good"
    else:
        category = "Polluted"

    # Confidence of predicted class
    confidence = probability[int(prediction)]

    return {
        "category": category,
        "confidence": round(float(confidence), 4)
    }
