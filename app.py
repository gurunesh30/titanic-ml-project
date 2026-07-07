from fastapi import FastAPI
from predictor import predict
from schemas import Passenger

# 1. Fixed typo: "veersion" -> "version"
app = FastAPI(title="Titanic Survival API", version="1.0")

@app.get("/")
def home():
    return {
        "message": "Titanic Prediction API is running"
    }   

@app.post("/predict")
def predict_survival(passenger: Passenger):
    result = predict(passenger)
    
    return result