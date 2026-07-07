import joblib
import pandas as pd 
from typing import Optional
from pydantic import BaseModel

# 1. Pydantic model for request validation
class Passenger(BaseModel):
    pclass: int
    sex: str
    age: Optional[float] = None
    fare: Optional[float] = None

# 2. Fixed path: Ensure this matches where joblib actually saved it
# Change "model/pipeline.pkl" to "models/pipeline.pkl"
pipeline = joblib.load("models/pipeline.pkl")

# 3. Enhanced function accepting the Pydantic schema
def predict(passenger: Passenger):
    # .model_dump() converts the Pydantic object into a clean dictionary
    df = pd.DataFrame([passenger.model_dump()])

    # Make predictions
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": {
            "No survival": probability[0],
            "Survival": probability[1]
        }
    }
