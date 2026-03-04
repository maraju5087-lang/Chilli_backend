import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("\n=========== UNIFIED TRAINING STARTED ===========")

# Load daily data
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

        data = df[[variety]].resample(rule).mean().dropna()

        data["year"] = data.index.year
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["day_of_week"] = data.index.dayofweek
        data["week_of_year"] = data.index.isocalendar().week.astype(int)

        data["lag_1"] = data[variety].shift(1)
        data["lag_2"] = data[variety].shift(2)
        data["roll_3"] = data[variety].rolling(3).mean()

        data = data.dropna()

        features = [
            "year","month","day",
            "day_of_week","week_of_year",
            "lag_1","lag_2","roll_3"
        ]

        X = data[features]
        y = data[variety]

        model = RandomForestRegressor(
            n_estimators=800,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        )

        # 🔹 Yearly → use full data (small dataset)
        if level_name == "yearly":

            model.fit(X, y)

            print("      Trained on full yearly data (no test split)")

        else:
            split = int(len(data) * 0.8)

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            print(f"      MAE: {mae:.2f} | R2: {r2:.4f}")

        joblib.dump(model, f"models/rf_{variety}_{level_name}.pkl")

print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY")