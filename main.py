from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import uvicorn
import google.generativeai as genai
import os

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# ================= GEMINI AI =================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

system_prompt = """
You are an expert agricultural assistant specialized in chilli farming.

Answer questions about:
- Chilli diseases
- Chilli cultivation
- Fertilizers
- Pest control
- Irrigation
- Market prices
- Chilli varieties

Give clear explanation and practical advice for farmers.
"""



# ================= IMAGE PREPROCESS =================
def preprocess_image(image):

    image = image.resize((224,224))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ================= CHAT MODEL =================
class ChatRequest(BaseModel):
    message: str


# ================= VARIETIES =================
varieties = ["byd_335", "teja", "g_274", "lca_334"]

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


# ================= CHATBOT =================
@app.post("/chat")
def chat_ai(data: ChatRequest):

    msg = data.message.lower()

    if "price" in msg:
        reply = "Current Guntur chilli price around ₹190000 per ton."

    elif "guntur" in msg:
        reply = "Guntur is the largest chilli market in Asia."

    elif "variety" in msg:
        reply = "Popular chilli varieties are Teja, BYD-335, LCA-334, and G-274."

    elif "hello" in msg or "hi" in msg:
        reply = "Hello! Ask me about chilli farming, diseases, fertilizers and market prices."

    else:

        try:

            response = model.generate_content(
                f"""
{system_prompt}

Farmer question:
{data.message}
"""
            )

            reply = ""

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        reply += part.text

            if reply == "":
                reply = "AI could not generate response."

        except Exception as e:

            print("GEMINI ERROR:", e)
            reply = "AI chatbot error. Please try again."

    return {"reply": reply}




# ================= LIVE MARKET PRICE =================
@app.get("/live_price")
def live_price():

    return {
        "market": "Guntur Market",
        "price": 190500,
        "unit": "₹ per ton"
    }
@app.get("/price_trend")
def get_price_trend(
        variety: str,
        year: int,
        month: int,
        week_from: int,
        week_to: int
):

    variety = variety.lower().replace("-", "_")

    df = pd.read_csv(BASE_DIR / "daily.csv")

    df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True)

    df = df.dropna(subset=["Dates"])

    df.set_index("Dates", inplace=True)

    series = df[[variety]].resample("D").mean().dropna()

    ml_model = load_model("rf", variety, "daily")

    days = []
    prices = []

    for week in range(week_from, week_to + 1):

        start_date = pd.to_datetime(
            f"{year}-W{week}-1",
            format="%G-W%V-%u"
        )

        for i in range(7):

            target_date = start_date + pd.Timedelta(days=i)

            if target_date in series.index:

                price = float(series.loc[target_date][variety])

            else:

                values = series[variety].tolist()
                last_date = series.index[-1]

                while last_date < target_date:

                    lag_1 = values[-1]
                    lag_2 = values[-2]
                    roll_3 = sum(values[-3:]) / 3

                    next_date = last_date + pd.Timedelta(days=1)

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

                price = values[-1]

            days.append(target_date.day)
            prices.append(price)

    return {
        "days": days,
        "prices": prices
    }
# ================= LOAD MODEL =================
def load_model(model_type: str, variety: str, level_name: str):

    if model_type == "rf":
        model_path = BASE_DIR / f"models/rf_{variety}_{level_name}.pkl"

    elif model_type == "prophet":
        model_path = BASE_DIR / f"models/prophet_{variety}_{level_name}.pkl"

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    return joblib.load(model_path)


# ================= PREDICTION =================
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

    try:
        df = pd.read_csv(BASE_DIR / "daily.csv")
    except:
        raise HTTPException(status_code=500, detail="daily.csv not found")

    df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Dates"])

    df.set_index("Dates", inplace=True)

    series = df[[variety]].resample(rule).mean().dropna()

    if level == "year":
        target_date = pd.Timestamp(year=year, month=1, day=1)

    elif level == "month":

        if month is None:
            raise HTTPException(status_code=400, detail="Month required")

        target_date = pd.Timestamp(year=year, month=month, day=1)

    elif level == "week":

        if week is None:
            raise HTTPException(status_code=400, detail="Week required")

        target_date = pd.to_datetime(
            f"{year}-W{int(week)}-1",
            format="%G-W%V-%u"
        )

    elif level == "day":

        if month is None or day is None:
            raise HTTPException(status_code=400, detail="Month and Day required")

        target_date = pd.Timestamp(year=year, month=month, day=day)

    else:
        raise HTTPException(status_code=400, detail="Invalid level")

    if target_date in series.index:

        return {
            "model": model.upper(),
            "variety": variety,
            "level": level,
            "predicted_price": float(series.loc[target_date][variety])
        }

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

            "year", "month", "day",
            "day_of_week", "week_of_year",
            "lag_1", "lag_2", "roll_3"

        ])

        next_price = float(ml_model.predict(X)[0])

        values.append(next_price)

        last_date = next_date

    return {
        "model": model.upper(),
        "variety": variety,
        "level": level,
        "predicted_price": round(values[-1], 2)
    }


# ================= RUN SERVER =================
if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)