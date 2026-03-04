from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Supported varieties
varieties = ["byd_335", "teja", "g_274", "lca_334"]

# Level mapping
levels_map = {
    "day": ("daily", "D"),
    "week": ("weekly", "W"),
    "month": ("monthly", "M"),
    "year": ("yearly", "Y")
}


# ================= HOME =================
@app.get("/")
def home():
    return {"message": "Unified Chilli ML Backend Running"}


# ================= LOAD MODEL =================
def load_model(model_type: str, variety: str, level_name: str):

    if model_type == "rf":
        path = BASE_DIR / f"models/rf_{variety}_{level_name}.pkl"
    elif model_type == "prophet":
        path = BASE_DIR / f"models/prophet_{variety}_{level_name}.pkl"
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return joblib.load(path)


# ================= PREDICT =================
@app.get("/predict")
def predict(
    model: str,
    variety: str,
    level: str,
    year: int,
    month: int = None,
    week: int = None,
    day: int = None
):

    model = model.lower()
    variety = variety.lower().replace("-", "_")
    level = level.lower()

    if variety not in varieties:
        raise HTTPException(status_code=400, detail="Invalid variety")

    if level not in levels_map:
        raise HTTPException(status_code=400, detail="Invalid level")

    level_name, rule = levels_map[level]

    ml_model = load_model(model, variety, level_name)

    df = pd.read_csv("daily.csv")
    df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Dates"])
    df.set_index("Dates", inplace=True)

    series = df[[variety]].resample(rule).mean().dropna()

    # ================= TARGET DATE LOGIC =================

    if level == "year":
        target_date = pd.Timestamp(year=year, month=1, day=1)

    elif level == "month":
        if month is None:
            raise HTTPException(status_code=400, detail="Month required")
        target_date = pd.Timestamp(year=year, month=month, day=1)

    elif level == "week":
        if week is None:
            raise HTTPException(status_code=400, detail="Week required")
        target_date = pd.to_datetime(f"{year}-W{int(week)}-1", format="%G-W%V-%u")

    elif level == "day":
        if month is None or day is None:
            raise HTTPException(status_code=400, detail="Month and Day required")
        target_date = pd.Timestamp(year=year, month=month, day=day)

    else:
        raise HTTPException(status_code=400, detail="Invalid level")

    # ================= HISTORICAL =================

    if target_date in series.index:
        return {
            "model": model.upper(),
            "variety": variety,
            "level": level,
            "predicted_price": float(series.loc[target_date][variety])
        }

    # ================= PROPHET =================

    if model == "prophet":

        future = pd.DataFrame({"ds": [target_date]})
        forecast = ml_model.predict(future)

        return {
            "model": "PROPHET",
            "variety": variety,
            "level": level,
            "predicted_price": round(float(forecast["yhat"].iloc[0]), 2)
        }

    # ================= RF RECURSIVE =================

    values = series[variety].tolist()
    last_date = series.index[-1]

    while last_date < target_date:

        lag_1 = values[-1]
        lag_2 = values[-2]
        roll_3 = sum(values[-3:]) / 3

        next_date = last_date + pd.tseries.frequencies.to_offset(rule)

        X = pd.DataFrame([[ 
            next_date.year,
            next_date.month,
            next_date.day,
            next_date.dayofweek,
            next_date.isocalendar().week,
            lag_1,
            lag_2,
            roll_3
        ]], columns=[
            "year","month","day",
            "day_of_week","week_of_year",
            "lag_1","lag_2","roll_3"
        ])

        next_price = float(ml_model.predict(X)[0])

        values.append(next_price)
        last_date = next_date

    return {
        "model": "RF",
        "variety": variety,
        "level": level,
        "predicted_price": round(values[-1], 2)
    }