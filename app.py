from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import joblib

# Initialize app
app = FastAPI(title="🏠 House Price Prediction API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
model = joblib.load("house_price_stack_pipeline.joblib")

# === 1. Define the input schema ===
class HouseFeatures(BaseModel):
    Id: int
    MSSubClass: int
    LotFrontage: float
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: float
    ExterQual: str
    BsmtQual: str
    BsmtFinSF1: float
    BsmtFinSF2: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    HeatingQC: str
    FirstFlrSF: float         # ← renamed from '1stFlrSF'
    SecondFlrSF: float        # ← renamed from '2ndFlrSF'
    LowQualFinSF: float
    GrLivArea: float
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Fireplaces: int
    FireplaceQu: str
    GarageYrBlt: float
    GarageCars: int
    GarageArea: float
    GarageQual: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int        # ← renamed from '3SsnPorch'
    ScreenPorch: int
    PoolArea: int
    MiscVal: int
    MoSold: int
    YrSold: int
    TotalSF: float
    HouseAge: int
    RemodelAge: int
    IsRemodeled: bool

# === 2. Define Kaggle column name mappings ===
FIELD_RENAMES = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "ThreeSsnPorch": "3SsnPorch"
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

    return {"predicted_price": round(price, 2)}

@app.get("/")
def root():
    return FileResponse("static/index.html")
