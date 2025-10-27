from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
import logging
import traceback
from pydantic import BaseModel, field_validator

logging.basicConfig(
    filename="api_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


model = joblib.load("model.pkl")
historical_df=pd.read_csv("historical_df.csv")
historical_df["Date"] = pd.to_datetime(historical_df["Date"])
historical_df = historical_df.sort_values("Date").tail(500)

app = FastAPI(
    title="Financial Return Prediction API",
    description="Predicts next-hour return from OHLCTV data",
    version="1.0"
)

class InputData(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Trades: float
    Volume: float

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/info")
def get_info():
    
    return {
        "model": "Random Forest Regression",
        "trained_on": "hourly OHLCTV data",
        "target": "next-hour return",
        "features_used": [
            "return_t-1", "return_t-5", "return_t-10",
            "sma_5", "sma_20", "ema_10",
            "volatility_24h", "price_ratio",
            "hour", "dayofweek", "volume_ratio",
            "momentum_3", "momentum_10"
        ],
        "usage": "Send POST to /predict with JSON containing Date, Open, High, Low, Close, Trades, Volume."
    }


@app.post("/predict")
def predict(data: InputData):
    global historical_df

    

    try:

        check_fields=["Open","High","Low","Close","Trades","Volume"]
        errors=[]

        for field in check_fields:
            value=getattr(data,field)
            if value<0:
                errors.append(f"{field} cannot be negative (got {value}).")

        if not (data.Low<=data.Close<=data.High):
            errors.append("Close must be between Low and High.")
        if any(getattr(data, f)==0 for f in ["Open","High","Low","Close"]):
            errors.append("Prices cannot be zero.")

        try:
            date_value = datetime.fromisoformat(data.Date)
            if date_value.tzinfo is None:
                date_value = date_value.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)

            if date_value < now.replace(year=now.year - 5):
                errors.append("Date is too old.")
        except ValueError:
            errors.append("Date must be in format: YYYY-MM-DDTHH:MM:SS")

        if errors:
            raise HTTPException(
                status_code=400,
                detail={"message":"Invalid input data.","errors":errors}
            )
        
        new_row = pd.DataFrame([data.model_dump()])
        new_row["Date"] = pd.to_datetime(new_row["Date"])
        historical_df = pd.concat([historical_df, new_row]).tail(500)
        
        df = historical_df.copy()
        df["return"] = df["Close"].pct_change()
        df["return_t-1"] = df["return"].shift(1)
        df["return_t-5"] = df["return"].shift(5)
        df["return_t-10"] = df["return"].shift(10)
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["volatility_10"] = df["return"].rolling(10).std()
        df["volatility_24h"] = df["return"].rolling(24).std()
        
        df["price_ratio"] = df.apply(lambda row: (row["Close"] / row["Open"]) if row["Open"] != 0 else np.nan, axis=1)
        
        df["hour"] = df["Date"].dt.hour
        df["dayofweek"] = df["Date"].dt.dayofweek
        
        df["volume_avg_10"] = df["Volume"].rolling(10).mean()
        df["volume_ratio"] = df.apply(lambda row: (row["Volume"] / row["volume_avg_10"]) if row["volume_avg_10"] != 0 else np.nan, axis=1)
        df.drop(columns=["volume_avg_10"], inplace=True)
        
        df["momentum_3"] = df["Close"] - df["Close"].shift(3)
        df["momentum_5"]=df["Close"]-df["Close"].shift(5)
        df["momentum_10"] = df["Close"] - df["Close"].shift(10)

        features = [
            "return_t-1", "return_t-5", "return_t-10",
            "sma_5", "sma_20", "ema_10", "volatility_10",
            "volatility_24h", "price_ratio", "hour",
            "dayofweek", "volume_ratio", "momentum_3","momentum_5", "momentum_10"
        ]

        latest = df.dropna(subset=features).iloc[-1:]

        if latest.empty:
            raise ValueError("Not enough historical data to compute features (NaN after drop).")

        prediction = model.predict(latest[features])[0]
        logging.info(f"Prediction successful for {data.Date}: {prediction:.6f}")

        return {
            "predicted_return": float(prediction),
            "predicted_percentage": f"{prediction * 100:.3f}%",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"Prediction failed: {e}\n{error_message}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



