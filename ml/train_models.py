"""Train all ML models from the CSVs in backend/data.

Run:  python -m ml.train_models        (from the backend folder)

Models
------
1. Crop yield model (XGBoost regressor)
   data: Indian_crop_production_yield_dataset.csv (~576k rows, 1997-2020)
   target: log(yield)   features: state, district, season, crop, year, log(area)
   evaluation: time split - train <= 2016, test 2017-2020, compared with the
   "historical average for this district+crop+season" baseline.

2. Price model (XGBoost regressor)
   data: Agriculture_price_dataset.csv (~737k daily mandi rows, 2023-2025)
         + mandi_data_2000_rows.csv (extra mandi observations)
   target: log(next-month price / current price)
   features: commodity, state, month, recent price lags and momentum
   evaluation: last 3 months held out, compared with "price stays the same".

3. Crop profile table: per crop, the seasons/states it is grown in, typical
   yield and area, and how often each district grows it (used to estimate
   "is this crop realistic here").

Everything is saved to backend/ml/artifacts/ with a metrics.json report.
"""

import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DATA = BACKEND / "data"
OUT = BACKEND / "ml" / "artifacts"


def key(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_cat(frame, columns, categories=None):
    categories = categories or {}
    for column in columns:
        cats = categories.get(column)
        if cats is None:
            cats = sorted(frame[column].dropna().unique().tolist())
            categories[column] = cats
        frame[column] = pd.Categorical(frame[column], categories=cats)
    return frame, categories


# --------------------------------------------------------------------------
# 1. Yield model
# --------------------------------------------------------------------------
YIELD_FEATURES = ["state", "district", "season", "crop", "year", "log_area"]
YIELD_CATS = ["state", "district", "season", "crop"]


def load_crop_data():
    df = pd.read_csv(DATA / "Indian_crop_production_yield_dataset.csv")
    df = df.rename(columns={
        "State_Name": "state", "District_Name": "district", "Crop_Year": "year",
        "Season": "season", "Crop": "crop_name", "Area": "area",
        "Production": "production", "yield": "yield",
    })
    df["state"] = df["state"].map(key)
    df["district"] = df["district"].map(key)
    df["season"] = df["season"].map(key)
    df["crop_name"] = df["crop_name"].astype(str).str.strip()
    df["crop"] = df["crop_name"].map(key)
    for column in ["area", "production", "yield", "year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["area", "yield", "year"])
    df = df[(df["area"] > 0) & (df["yield"] > 0)]

    # Remove data-entry outliers: outside the 1st-99th percentile per crop.
    bounds = df.groupby("crop")["yield"].quantile([0.01, 0.99]).unstack()
    df = df.join(bounds, on="crop")
    df = df[(df["yield"] >= df[0.01]) & (df["yield"] <= df[0.99])].drop(columns=[0.01, 0.99])
    df["log_area"] = np.log1p(df["area"])
    df["log_yield"] = np.log(df["yield"])
    return df.reset_index(drop=True)


def train_yield_model(df):
    train = df[df["year"] <= 2016].copy()
    test = df[df["year"] > 2016].copy()

    # Baseline: historical mean log-yield for district+crop+season (fallback crop mean).
    hist = train.groupby(["district", "crop", "season"])["log_yield"].mean()
    crop_mean = train.groupby("crop")["log_yield"].mean()
    base = test.join(hist.rename("b1"), on=["district", "crop", "season"]).join(crop_mean.rename("b2"), on="crop")
    baseline_pred = base["b1"].fillna(base["b2"]).fillna(train["log_yield"].mean())

    train, cats = to_cat(train, YIELD_CATS)
    test, _ = to_cat(test, YIELD_CATS, cats)

    params = dict(
        n_estimators=1200, learning_rate=0.05, max_depth=9, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse",
    )
    model = xgb.XGBRegressor(**params)
    model.fit(train[YIELD_FEATURES], train["log_yield"], eval_set=[(test[YIELD_FEATURES], test["log_yield"])], verbose=False)
    pred = model.predict(test[YIELD_FEATURES])

    def scores(y, p):
        y, p = np.asarray(y), np.asarray(p)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        # median absolute % error on the real yield scale
        mdape = float(np.median(np.abs(np.exp(p) - np.exp(y)) / np.exp(y)) * 100)
        return {"r2_log": round(float(r2), 4), "median_abs_pct_error": round(mdape, 2)}

    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split": "train <= 2016, test 2017-2020",
        "model": scores(test["log_yield"], pred),
        "baseline_district_crop_mean": scores(test["log_yield"], baseline_pred),
        "best_iteration": int(model.best_iteration),
    }

    # Refit on all years with the chosen number of trees for production use.
    full, cats = to_cat(df.copy(), YIELD_CATS)
    params.pop("early_stopping_rounds")
    params["n_estimators"] = int(model.best_iteration) + 1
    final = xgb.XGBRegressor(**params)
    final.fit(full[YIELD_FEATURES], full["log_yield"], verbose=False)
    return final, cats, metrics


def build_crop_profiles(df):
    recent = df[df["year"] >= df["year"].max() - 9]
    profiles = {}
    for crop, group in df.groupby("crop"):
        profiles[crop] = {
            "name": group["crop_name"].mode().iat[0],
            "seasons": group["season"].value_counts(normalize=True).round(3).to_dict(),
            "median_yield": float(group["yield"].median()),
            "median_area": float(group["area"].median()),
            "states": int(group["state"].nunique()),
            "records": int(len(group)),
        }
    # How often each district grew each crop in each season (last 10 years).
    presence = (
        recent.groupby(["state", "district", "season", "crop"]).agg(
            years=("year", "nunique"), area=("area", "median"), yield_median=("yield", "median")
        ).reset_index()
    )
    district_years = recent.groupby(["state", "district"])["year"].nunique().rename("district_years")
    presence = presence.join(district_years, on=["state", "district"])
    presence["frequency"] = (presence["years"] / presence["district_years"]).clip(0, 1).round(3)
    locations = (
        df.groupby("state")["district"].apply(lambda s: sorted(s.unique().tolist())).to_dict()
    )
    return profiles, presence, locations


# --------------------------------------------------------------------------
# 2. Price model
# --------------------------------------------------------------------------
PRICE_FEATURES = ["commodity", "state", "month", "lag0", "r1", "r2", "r3", "roll3", "vol3", "n_obs"]
PRICE_CATS = ["commodity", "state"]


def load_price_monthly():
    big = pd.read_csv(
        DATA / "Agriculture_price_dataset.csv",
        usecols=["STATE", "District Name", "Commodity", "Modal_Price", "Price Date"],
    ).rename(columns={"STATE": "state", "District Name": "district", "Commodity": "commodity",
                      "Modal_Price": "price", "Price Date": "date"})
    big["date"] = pd.to_datetime(big["date"], format="%m/%d/%Y", errors="coerce")

    small = pd.read_csv(DATA / "mandi_data_2000_rows.csv").rename(columns={
        "State": "state", "District": "district", "Commodity": "commodity",
        "Modal Price": "price", "Date": "date"})[["state", "district", "commodity", "price", "date"]]
    small["date"] = pd.to_datetime(small["date"], format="%m/%d/%Y", errors="coerce")

    df = pd.concat([big, small], ignore_index=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    for column in ["state", "district", "commodity"]:
        df[column] = df[column].map(key)
    df["month_p"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["commodity", "state", "district", "month_p"])["price"]
        .agg(price="median", n_obs="size").reset_index()
    )
    return monthly


def make_price_features(monthly):
    rows = []
    for (commodity, state, district), g in monthly.groupby(["commodity", "state", "district"]):
        g = g.set_index("month_p").sort_index()
        full = pd.period_range(g.index.min(), g.index.max(), freq="M")
        g = g.reindex(full)
        g["price"] = g["price"].interpolate(limit=2)
        g["n_obs"] = g["n_obs"].fillna(0)
        logp = np.log(g["price"])
        frame = pd.DataFrame({
            "commodity": commodity, "state": state, "district": district,
            "period": g.index, "month": g.index.month,
            "lag0": logp,
            "r1": logp - logp.shift(1), "r2": logp - logp.shift(2), "r3": logp - logp.shift(3),
            "roll3": logp - logp.rolling(3).mean(),
            "vol3": logp.diff().rolling(3).std(),
            "n_obs": np.log1p(g["n_obs"]),
            "target": logp.shift(-1) - logp,
        })
        rows.append(frame)
    feats = pd.concat(rows, ignore_index=True)
    return feats.dropna(subset=["lag0", "r1"])


def train_price_model(monthly):
    feats = make_price_features(monthly)
    labelled = feats.dropna(subset=["target"])
    last = labelled["period"].max()
    cutoff = last - 3
    train = labelled[labelled["period"] <= cutoff].copy()
    test = labelled[labelled["period"] > cutoff].copy()

    train, cats = to_cat(train, PRICE_CATS)
    test, _ = to_cat(test, PRICE_CATS, cats)
    params = dict(
        n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", reg_lambda=5.0,
        enable_categorical=True, n_jobs=-1, early_stopping_rounds=40, eval_metric="rmse",
    )
    model = xgb.XGBRegressor(**params)
    model.fit(train[PRICE_FEATURES], train["target"], eval_set=[(test[PRICE_FEATURES], test["target"])], verbose=False)
    pred = model.predict(test[PRICE_FEATURES])

    actual = np.exp(test["lag0"] + test["target"])
    model_price = np.exp(test["lag0"] + pred)
    naive_price = np.exp(test["lag0"])
    mape = lambda p: round(float(np.median(np.abs(p - actual) / actual) * 100), 2)
    direction = float(np.mean(np.sign(pred) == np.sign(test["target"])) * 100)
    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split": f"train <= {cutoff}, test {cutoff + 1}..{last} (1-month-ahead)",
        "model_median_abs_pct_error": mape(model_price),
        "naive_median_abs_pct_error": mape(naive_price),
        "direction_accuracy_pct": round(direction, 1),
        "best_iteration": int(model.best_iteration),
    }

    full, cats = to_cat(labelled.copy(), PRICE_CATS)
    params.pop("early_stopping_rounds")
    params["n_estimators"] = int(model.best_iteration) + 1
    final = xgb.XGBRegressor(**params)
    final.fit(full[PRICE_FEATURES], full["target"], verbose=False)

    # Latest feature row per series, so the API can predict without recomputing.
    latest = feats.sort_values("period").groupby(["commodity", "state", "district"]).tail(1)
    return final, cats, metrics, latest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    t = time.time()
    crops = load_crop_data()
    yield_model, yield_cats, report["yield_model"] = train_yield_model(crops)
    profiles, presence, locations = build_crop_profiles(crops)
    joblib.dump({"model": yield_model, "categories": yield_cats, "features": YIELD_FEATURES,
                 "max_year": int(crops["year"].max())}, OUT / "yield_model.joblib")
    joblib.dump({"profiles": profiles, "presence": presence, "locations": locations}, OUT / "crop_profiles.joblib")
    print(f"Yield model done in {time.time() - t:.0f}s: {report['yield_model']}")

    t = time.time()
    monthly = load_price_monthly()
    price_model, price_cats, report["price_model"], latest = train_price_model(monthly)
    joblib.dump({"model": price_model, "categories": price_cats, "features": PRICE_FEATURES,
                 "latest": latest}, OUT / "price_model.joblib")
    print(f"Price model done in {time.time() - t:.0f}s: {report['price_model']}")

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved models and metrics to {OUT}")


if __name__ == "__main__":
    main()
