from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="🏡 House Price Predictor API")

# Load the trained pipeline
model = joblib.load("house_price_stack_pipeline.joblib")

class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int
    TotalBsmtSF: float
    GarageCars: int

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Predictor API!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_data = np.array([[features.GrLivArea, features.OverallQual,
                            features.TotalBsmtSF, features.GarageCars]])
    log_price = model.predict(input_data)[0]
    price = np.expm1(log_price)
    return {"predicted_price": round(price, 2)}
