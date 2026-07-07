import fastapi from Fastapi 

from predictor import predict
from schemas import passenger

app = Fastapi(title="Titanic Survival API",veersion="1.0")

@app.get("/")
def home():
    return {
        "message":"Titanic Prediction API is running"
    }   