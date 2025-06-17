from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Initialize app
app = FastAPI(title="🏠 House Price Prediction API")

# Load trained model
model = joblib.load("house_price_stack_pipeline.joblib")

# === 1. Define the input schema ===
class HouseFeatures(BaseModel):
    GrLivArea: float
    TotalBsmtSF: float
    GarageCars: int
    GarageArea: float
    LotArea: int
    OverallQual: int
    YearBuilt: int
    FirstFlrSF: float  # ← maps to "1stFlrSF"
    SecondFlrSF: float  # ← maps to "2ndFlrSF"
    TotRmsAbvGrd: int
    BsmtFullBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    Fireplaces: int
    WoodDeckSF: int
    OpenPorchSF: int
    MoSold: int
    YrSold: int

    # Categorical
    Neighborhood: str
    ExterQual: str
    BsmtQual: str
    KitchenQual: str
    FireplaceQu: str
    GarageQual: str
    HeatingQC: str
    HouseStyle: str
    RoofStyle: str
    RoofMatl: str
    MSZoning: str
    SaleCondition: str
    Exterior1st: str
    Exterior2nd: str

# === 2. Define Kaggle column name mappings ===
FIELD_RENAMES = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF"
    }

# === 3. Prediction endpoint ===
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert to dict
    data = features.dict()

    # Rename fields back to original Kaggle column names
    for py_key, kaggle_key in FIELD_RENAMES.items():
        data[kaggle_key] = data.pop(py_key)

    # Create a DataFrame with one row
    input_df = pd.DataFrame([data])

    # Predict log price and convert back
    log_price = model.predict(input_df)[0]
    price = np.expm1(log_price)

    return {"predicted_price": float(price)}
