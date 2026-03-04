import pandas as pd
from prophet import Prophet
import joblib
import os

print("\n=========== UNIFIED PROPHET TRAINING STARTED ===========")

# Load daily data (SINGLE SOURCE)
df = pd.read_csv("daily.csv")
df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Dates"])
df = df.sort_values("Dates")
df.set_index("Dates", inplace=True)

varieties = ["byd_335", "teja", "g_274", "lca_334"]
levels = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "yearly": "YE"
}

os.makedirs("models", exist_ok=True)

for variety in varieties:
    print(f"\n🔁 Processing {variety}")

    for level_name, rule in levels.items():

        print(f"   → Training {level_name}")

        # Resample from single daily source
        data = df[[variety]].resample(rule).mean().dropna()

        # Prophet requires columns ds & y
        prophet_df = data.reset_index()
        prophet_df.columns = ["ds", "y"]

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(prophet_df)

        joblib.dump(model, f"models/prophet_{variety}_{level_name}.pkl")

        print(f"      ✅ Saved prophet_{variety}_{level_name}.pkl")

print("\n🎉 ALL PROPHET MODELS TRAINED SUCCESSFULLY")