"""Crop recommendation with the trained ML models + live weather.

Uses the artifacts produced by `python -m ml.train_models`:
  * yield_model.joblib    XGBoost yield regressor (state, district, season, crop, year, area)
  * price_model.joblib    XGBoost next-month mandi price model + latest price per series
  * crop_profiles.joblib  how often each district grows each crop, typical yields

Every candidate crop gets five sub-scores (0..1), combined into one ranking:
  local   how regularly this district grows the crop in this season (dataset)
  weather fit of the live OpenWeather 5-day outlook to the crop's temperature/rain needs
  soil    fit of the farmer's soil, water source and irrigation to the crop
  yield   ML-predicted yield for this district relative to the district's own history
  revenue predicted yield x price (ML price model, else MSP / reference price)
"""

import difflib
import math
import re
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import settings

ACRE_TO_HA = 0.404686

# Agronomy reference per crop key (dataset crop names, normalised).
#   temp: comfortable mean air temperature (deg C) during growth
#   water: low / medium / high requirement
#   soils: soils where the crop does well
#   cost: cultivation cost tier (Rs per acre roughly: low <15k, medium 15-40k, high >40k)
#   price: MSP 2024-25 / 2025-26 or approximate mandi reference, Rs per quintal
#   ref_yield: typical India yield, quintal/ha (the dataset's yield units differ
#              between states, so the model output is used as a ratio to the
#              district's own history and applied to this reference)
#   disease: susceptible to fungal disease in warm humid weather
CROP_INFO = {
    "rice":            dict(ref_yield=40, temp=(20, 35), water="high", soils={"clay", "alluvial", "loamy", "silt"}, cost="medium", price=2300, msp=True),
    "wheat":           dict(ref_yield=35, temp=(10, 25), water="medium", soils={"loamy", "alluvial", "clay", "black", "silt"}, cost="medium", price=2425, msp=True),
    "maize":           dict(ref_yield=30, temp=(18, 32), water="medium", soils={"loamy", "alluvial", "red", "black", "silt"}, cost="medium", price=2225, msp=True),
    "jowar":           dict(ref_yield=11, temp=(20, 35), water="low", soils={"black", "loamy", "red", "alluvial"}, cost="low", price=3371, msp=True),
    "bajra":           dict(ref_yield=15, temp=(22, 36), water="low", soils={"sandy", "red", "loamy", "black"}, cost="low", price=2625, msp=True),
    "ragi":            dict(ref_yield=16, temp=(18, 30), water="low", soils={"red", "laterite", "loamy", "sandy"}, cost="low", price=4290, msp=True),
    "barley":          dict(ref_yield=30, temp=(8, 24), water="low", soils={"loamy", "sandy", "alluvial"}, cost="low", price=1980, msp=True),
    "small millets":   dict(ref_yield=9, temp=(18, 32), water="low", soils={"red", "sandy", "laterite"}, cost="low", price=3500),
    "gram":            dict(ref_yield=12, temp=(12, 28), water="low", soils={"black", "loamy", "alluvial", "clay"}, cost="low", price=5650, msp=True),
    "arhar tur":       dict(ref_yield=9, temp=(20, 32), water="low", soils={"black", "loamy", "red", "alluvial"}, cost="low", price=7550, msp=True),
    "moong green gram": dict(ref_yield=6, temp=(22, 34), water="low", soils={"loamy", "sandy", "red", "alluvial"}, cost="low", price=8682, msp=True),
    "urad":            dict(ref_yield=6, temp=(22, 34), water="low", soils={"loamy", "black", "clay", "alluvial"}, cost="low", price=7400, msp=True),
    "masoor":          dict(ref_yield=11, temp=(10, 25), water="low", soils={"loamy", "alluvial", "clay"}, cost="low", price=6700, msp=True),
    "lentil":          dict(ref_yield=11, temp=(10, 25), water="low", soils={"loamy", "alluvial", "clay"}, cost="low", price=6700, msp=True),
    "horse gram":      dict(ref_yield=5, temp=(20, 32), water="low", soils={"red", "laterite", "sandy"}, cost="low", price=5000),
    "cowpea lobia":    dict(ref_yield=7, temp=(20, 35), water="low", soils={"sandy", "loamy", "red"}, cost="low", price=6000),
    "groundnut":       dict(ref_yield=18, temp=(22, 33), water="medium", soils={"sandy", "red", "loamy"}, cost="medium", price=6783, msp=True),
    "soyabean":        dict(ref_yield=11, temp=(20, 32), water="medium", soils={"black", "loamy", "clay"}, cost="medium", price=4892, msp=True),
    "rapeseed mustard": dict(ref_yield=15, temp=(10, 25), water="low", soils={"loamy", "alluvial", "sandy"}, cost="low", price=5950, msp=True),
    "sunflower":       dict(ref_yield=8, temp=(18, 30), water="medium", soils={"black", "loamy", "red", "alluvial"}, cost="medium", price=7280, msp=True),
    "sesamum":         dict(ref_yield=5, temp=(22, 34), water="low", soils={"sandy", "loamy", "red", "black"}, cost="low", price=9267, msp=True),
    "safflower":       dict(ref_yield=8, temp=(15, 28), water="low", soils={"black", "loamy"}, cost="low", price=5940, msp=True),
    "linseed":         dict(ref_yield=6, temp=(10, 25), water="low", soils={"black", "loamy", "clay"}, cost="low", price=6000),
    "castor seed":     dict(ref_yield=18, temp=(20, 33), water="low", soils={"sandy", "red", "loamy"}, cost="low", price=6000),
    "niger seed":      dict(ref_yield=4, temp=(18, 28), water="low", soils={"red", "laterite", "loamy"}, cost="low", price=8717, msp=True),
    "cotton lint":     dict(ref_yield=None, temp=(21, 33), water="medium", soils={"black", "alluvial", "loamy", "red"}, cost="high", price=None, disease=True),
    "sugarcane":       dict(ref_yield=800, temp=(20, 35), water="high", soils={"loamy", "alluvial", "black", "clay"}, cost="high", price=340, msp=True),
    "potato":          dict(ref_yield=250, temp=(12, 24), water="medium", soils={"loamy", "sandy", "alluvial", "silt"}, cost="high", price=1500, disease=True),
    "onion":           dict(ref_yield=180, temp=(13, 28), water="medium", soils={"loamy", "alluvial", "black", "red"}, cost="high", price=2000, disease=True),
    "tomato":          dict(ref_yield=250, temp=(18, 29), water="medium", soils={"loamy", "red", "sandy", "black"}, cost="high", price=1500, disease=True),
    "brinjal":         dict(ref_yield=180, temp=(20, 32), water="medium", soils={"loamy", "silt", "alluvial", "black"}, cost="medium", price=1500),
    "cabbage":         dict(ref_yield=230, temp=(12, 24), water="medium", soils={"loamy", "clay", "alluvial"}, cost="medium", price=1000, disease=True),
    "cauliflower":     dict(ref_yield=190, temp=(12, 24), water="medium", soils={"loamy", "clay", "alluvial"}, cost="medium", price=1500, disease=True),
    "bhindi":          dict(ref_yield=120, temp=(22, 35), water="medium", soils={"loamy", "sandy", "alluvial"}, cost="medium", price=2000),
    "garlic":          dict(ref_yield=55, temp=(12, 25), water="medium", soils={"loamy", "silt", "clay"}, cost="high", price=8000),
    "dry chillies":    dict(ref_yield=25, temp=(20, 32), water="medium", soils={"black", "loamy", "red"}, cost="medium", price=15000, disease=True),
    "coriander":       dict(ref_yield=10, temp=(15, 28), water="low", soils={"loamy", "black", "silt"}, cost="low", price=7000),
    "turmeric":        dict(ref_yield=60, temp=(20, 32), water="high", soils={"loamy", "red", "clay", "laterite"}, cost="high", price=12000),
    "ginger":          dict(ref_yield=150, temp=(20, 30), water="high", soils={"loamy", "laterite", "red"}, cost="high", price=4000),
    "banana":          dict(ref_yield=350, temp=(20, 35), water="high", soils={"loamy", "alluvial", "clay"}, cost="high", price=1500),
    "sweet potato":    dict(ref_yield=120, temp=(20, 30), water="medium", soils={"sandy", "loamy", "red"}, cost="medium", price=1500),
    "tapioca":         dict(ref_yield=350, temp=(22, 35), water="medium", soils={"laterite", "red", "sandy", "loamy"}, cost="medium", price=1200),
    "jute":            dict(ref_yield=25, temp=(24, 35), water="high", soils={"alluvial", "loamy", "clay"}, cost="medium", price=5335, msp=True),
}

# Price model commodity name for a dataset crop key.
PRICE_MODEL_NAMES = {"rice": "rice", "wheat": "wheat", "maize": "maize", "potato": "potato",
                     "onion": "onion", "tomato": "tomato", "sugarcane": "sugarcane"}

SOIL_ALIASES = {"black regur": "black", "regur": "black", "black cotton": "black", "red sandy": "red"}
SEASON_MAP = {
    "rainy": ["kharif", "autumn"], "kharif": ["kharif", "autumn"], "monsoon": ["kharif", "autumn"],
    "winter": ["rabi", "winter"], "rabi": ["rabi", "winter"],
    "summer": ["summer"], "zaid": ["summer"], "whole year": ["whole year"],
}
SEASON_LABEL = {"kharif": "Kharif", "rabi": "Rabi", "summer": "Summer / Zaid", "whole year": "Whole Year"}
BUDGET_PER_ACRE = {"low": 12000, "medium": 25000, "high": 50000}


def key(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def season_from_month(month=None):
    month = month or datetime.now().month
    if 6 <= month <= 9:
        return "rainy"
    if month >= 10 or month <= 2:
        return "winter"
    return "summer"


class MLService:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self.error = None

    # ------------------------------------------------------------------ loading
    def load(self):
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            try:
                import joblib

                t = time.time()
                art = settings.ML_DIR
                profiles = joblib.load(art / "crop_profiles.joblib")
                yield_pack = joblib.load(art / "yield_model.joblib")
                price_pack = joblib.load(art / "price_model.joblib")
                self.profiles = profiles["profiles"]
                self.presence = profiles["presence"]
                self.locations = profiles["locations"]
                self.yield_model = yield_pack["model"]
                self.yield_cats = yield_pack["categories"]
                self.yield_features = yield_pack["features"]
                self.max_year = yield_pack["max_year"]
                self.price_model = price_pack["model"]
                self.price_cats = price_pack["categories"]
                self.price_features = price_pack["features"]
                self.price_latest = price_pack["latest"]
                self.all_districts = {d: s for s, ds in self.locations.items() for d in ds}
                try:
                    import json
                    self.metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
                except Exception:
                    self.metrics = {}
                self._loaded = True
                print(f"ML models loaded in {time.time() - t:.1f}s")
            except Exception as exc:
                self.error = str(exc)
                print(f"ML models unavailable: {exc}")
        return self._loaded

    @property
    def ready(self):
        return self._loaded

    # --------------------------------------------------------------- location
    def resolve_location(self, state, district):
        """Map free-text state/district to dataset keys (fuzzy)."""
        state_k, district_k = key(state), key(district)
        states = list(self.locations)
        if state_k not in self.locations and state_k:
            match = difflib.get_close_matches(state_k, states, n=1, cutoff=0.7)
            state_k = match[0] if match else state_k
        if district_k:
            pool = self.locations.get(state_k) or list(self.all_districts)
            if district_k not in pool:
                match = difflib.get_close_matches(district_k, pool, n=1, cutoff=0.75)
                district_k = match[0] if match else district_k
            if not state_k or state_k not in self.locations:
                state_k = self.all_districts.get(district_k, state_k)
        known_district = district_k in self.locations.get(state_k, [])
        return state_k, (district_k if known_district else None)

    def find_location_in_text(self, text):
        """Find a district (and its state) mentioned in free text."""
        if not self.load():
            return None, None
        t = f" {key(text)} "
        best = None
        for district, state in self.all_districts.items():
            if len(district) >= 4 and f" {district} " in t and (best is None or len(district) > len(best[0])):
                best = (district, state)
        if best:
            return best[1], best[0]
        for state in self.locations:
            if f" {state} " in t:
                return state, None
        return None, None

    def find_crop_in_text(self, text):
        t = f" {key(text)} "
        aliases = {"paddy": "rice", "dhan": "rice", "chana": "gram", "tur": "arhar tur", "arhar": "arhar tur",
                   "moong": "moong green gram", "mustard": "rapeseed mustard", "soybean": "soyabean",
                   "cotton": "cotton lint", "okra": "bhindi", "chilli": "dry chillies", "chilly": "dry chillies",
                   "sesame": "sesamum", "til": "sesamum", "millet": "bajra", "sorghum": "jowar", "cane": "sugarcane"}
        for alias, crop in aliases.items():
            if f" {alias} " in t:
                return crop
        for crop in sorted(CROP_INFO, key=len, reverse=True):
            if f" {crop} " in t:
                return crop
        return None

    # ------------------------------------------------------------------ models
    def _predict_yield(self, rows):
        frame = pd.DataFrame(rows)
        for column in ["state", "district", "season", "crop"]:
            frame[column] = pd.Categorical(frame[column], categories=self.yield_cats[column])
        frame["year"] = self.max_year
        return np.exp(self.yield_model.predict(frame[self.yield_features]))

    def price_outlook(self, crop, state=None, district=None):
        """Latest mandi price + ML next-month forecast, or MSP/reference price."""
        info = CROP_INFO.get(crop, {})
        commodity = PRICE_MODEL_NAMES.get(crop)
        if commodity and self.load():
            latest = self.price_latest[self.price_latest["commodity"] == commodity]
            scope = "India"
            if state:
                local = latest[latest["state"] == state]
                if district is not None and len(local):
                    near = local[local["district"] == district]
                    if len(near):
                        local, scope = near, district.title()
                    else:
                        scope = state.title()
                elif len(local):
                    scope = state.title()
                if len(local):
                    latest = local
            if len(latest):
                feats = latest.copy()
                for column in ["commodity", "state"]:
                    feats[column] = pd.Categorical(feats[column], categories=self.price_cats[column])
                change = self.price_model.predict(feats[self.price_features])
                now_price = float(np.median(np.exp(latest["lag0"])))
                next_price = float(np.median(np.exp(latest["lag0"].to_numpy() + change)))
                ref = info.get("price")
                if ref and not (0.6 <= next_price / ref <= 1.6):
                    # the stored model is trained on a small mandi sample; trust MSP / reference
                    # prices when it is far away from them (e.g. sugarcane FRP is ~Rs 340/q)
                    next_price = now_price = None
                if next_price is None:
                    return {"price": ref, "current": ref, "change_pct": 0.0, "as_of": "2025-26", "scope": "India",
                            "source": "Govt. MSP / FRP" if info.get("msp") else "reference mandi price (approx.)"}
                return {
                    "price": round(next_price), "current": round(now_price),
                    "change_pct": round((next_price / now_price - 1) * 100, 1),
                    "as_of": str(latest["period"].max()), "scope": scope,
                    "source": "XGBoost mandi price model", "series": int(len(latest)),
                }
        if info.get("price"):
            return {"price": info["price"], "current": info["price"], "change_pct": 0.0, "as_of": "2025-26",
                    "scope": "India", "source": "Govt. MSP" if info.get("msp") else "reference mandi price (approx.)"}
        return None

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _weather_score(info, weather, water_source, season_keys):
        if not weather:
            return 0.6, []
        outlook = weather.get("outlook", {})
        temp = outlook.get("temp_avg", weather.get("current", {}).get("temp"))
        notes = []
        lo, hi = info["temp"]
        if temp is None:
            score = 0.6
        elif lo <= temp <= hi:
            score = 1.0
        else:
            gap = lo - temp if temp < lo else temp - hi
            score = max(0.0, 1.0 - 0.09 * gap)
            notes.append(f"current temps (~{temp:.0f} C) are {'below' if temp < lo else 'above'} its {lo}-{hi} C comfort range")
        rain = outlook.get("total_rain_mm", 0) or 0
        rainfed = water_source in {"rainfed", ""}
        if info["water"] == "high" and rainfed and rain < 20 and "kharif" not in season_keys:
            score -= 0.3
            notes.append("needs a lot of water but the forecast is dry and the farm is rainfed")
        elif info["water"] == "low" and rain > 80:
            score -= 0.1
            notes.append("heavy rain forecast can waterlog this crop")
        if info.get("disease") and outlook.get("avg_humidity", 0) >= 85 and outlook.get("temp_max", 0) >= 25:
            score -= 0.15
            notes.append("warm humid weather raises fungal disease risk")
        return max(0.0, min(1.0, score)), notes

    @staticmethod
    def _soil_score(info, soil, water_source, irrigation, budget_per_acre):
        score, notes = 0.6, []
        if soil:
            if soil in info["soils"]:
                score = 1.0
            else:
                score = 0.35
                notes.append(f"{soil} soil is not ideal")
        irrigated = water_source not in {"", "rainfed"}
        if info["water"] == "high" and not irrigated:
            score -= 0.25
            notes.append("high water need on a rainfed farm")
        if irrigation == "drip" and info["cost"] == "high":
            score += 0.1
        if budget_per_acre:
            need = BUDGET_PER_ACRE[info["cost"]]
            if budget_per_acre < need * 0.6:
                score -= 0.3
                notes.append(f"usually needs about Rs {need:,}/acre, more than your budget per acre")
            elif budget_per_acre < need:
                score -= 0.12
        return max(0.0, min(1.0, score)), notes

    def recommend(self, payload, weather=None, top_n=5, soil_probs=None, live_prices=None):
        """Rank crops for one farm.

        soil_probs: optional {crop_key: probability} from the soil/NPK Random Forest
        model. When given, it becomes a sixth score so the farmer's soil test
        directly influences the ranking.
        live_prices: optional {crop_key: price dict} from today's mandi prices; used
        for the revenue score instead of the stored price model / MSP.
        """
        if not self.load():
            raise RuntimeError(f"ML models are not available: {self.error}")

        state, district = self.resolve_location(payload.get("stateName"), payload.get("districtName"))
        season_in = key(payload.get("season")) or season_from_month()
        season_keys = SEASON_MAP.get(season_in, SEASON_MAP[season_from_month()])
        soil = key(payload.get("soilType"))
        soil = SOIL_ALIASES.get(soil, soil.split(" ")[0] if soil else "")
        water_source = key(payload.get("waterSource"))
        irrigation = key(payload.get("irrigationType"))
        land_acres = float(payload.get("landSize") or 0) or None
        budget = float(payload.get("budget") or 0) or None
        budget_per_acre = budget / land_acres if budget and land_acres else None
        targeted = self.find_crop_in_text(payload.get("targetedCrop") or "")

        presence = self.presence
        local = presence[(presence["state"] == state) & (presence["district"] == district)] if district else presence.iloc[0:0]
        scope = "district"
        if local.empty:
            local = presence[presence["state"] == state]
            scope = "state"
        in_season = local[local["season"].isin(season_keys + ["whole year"])]

        candidates = {}
        for crop in CROP_INFO:
            rows = in_season[in_season["crop"] == crop]
            if crop == "rice" and rows.empty:
                rows = in_season[in_season["crop"] == "paddy"]
            if rows.empty:
                continue
            if scope == "state":
                # share of the state's districts growing it this season
                n_districts = max(local["district"].nunique(), 1)
                freq = min(1.0, rows["district"].nunique() / n_districts * 1.5)
                best = rows.sort_values("area", ascending=False).iloc[0]
            else:
                best = rows.sort_values(["frequency", "area"], ascending=False).iloc[0]
                freq = float(best["frequency"])
            candidates[crop] = {"freq": freq, "season": best["season"], "district": best["district"],
                                "area": float(best["area"]), "hist_yield": float(best["yield_median"])}
        if targeted and targeted not in candidates and targeted in self.profiles:
            prof = self.profiles[targeted]
            candidates[targeted] = {"freq": 0.0, "season": max(prof["seasons"], key=prof["seasons"].get),
                                    "district": district, "area": prof["median_area"], "hist_yield": prof["median_yield"]}
        if not candidates:
            raise RuntimeError(f"No crop history found for {payload.get('districtName')}, {payload.get('stateName')}.")

        crops = list(candidates)
        preds = self._predict_yield([{
            "state": state, "district": candidates[c]["district"], "season": candidates[c]["season"],
            "crop": c, "log_area": math.log1p(candidates[c]["area"]),
        } for c in crops])

        results = []
        for crop, pred in zip(crops, preds):
            info, cand = CROP_INFO[crop], candidates[crop]
            hist = cand["hist_yield"] or pred
            yield_ratio = float(np.clip(pred / hist, 0.6, 1.5)) if hist else 1.0
            yield_q_ha = info["ref_yield"] * yield_ratio if info["ref_yield"] else None
            w_score, w_notes = self._weather_score(info, weather, water_source, season_keys)
            s_score, s_notes = self._soil_score(info, soil, water_source, irrigation, budget_per_acre)
            # live mandi price (today, from the Mandi Prices API) beats the stored model / MSP price
            price = (live_prices or {}).get(crop) or self.price_outlook(crop, state, district)
            revenue_ha = None
            if price and yield_q_ha:
                revenue_ha = yield_q_ha * price["price"]
            results.append({
                "crop": crop, "info": info, "cand": cand, "yield_q_ha": yield_q_ha,
                "yield_ratio": float(yield_ratio), "weather": w_score, "soil": s_score,
                "notes": w_notes + s_notes, "price": price, "revenue_ha": revenue_ha,
            })

        revenues = sorted(r["revenue_ha"] for r in results if r["revenue_ha"])
        for r in results:
            if r["revenue_ha"] and len(revenues) > 1:
                r["revenue_score"] = revenues.index(r["revenue_ha"]) / (len(revenues) - 1)
            else:
                r["revenue_score"] = 0.5
            r["yield_score"] = max(0.0, min(1.0, 0.5 + (r["yield_ratio"] - 1) * 1.5))
            r["local"] = r["cand"]["freq"]
            r["score"] = (0.30 * r["local"] + 0.25 * r["weather"] + 0.15 * r["soil"]
                          + 0.10 * r["yield_score"] + 0.20 * r["revenue_score"])
            if soil_probs:
                top_prob = max(soil_probs.values()) or 1.0
                r["npk"] = min(1.0, soil_probs.get(r["crop"], 0.0) / top_prob)
                r["score"] = 0.85 * r["score"] + 0.15 * r["npk"]
            if targeted == r["crop"]:
                r["score"] += 0.05

        results.sort(key=lambda r: r["score"], reverse=True)
        top = results[:top_n]
        if targeted and all(r["crop"] != targeted for r in top):
            target_row = next((r for r in results if r["crop"] == targeted), None)
        else:
            target_row = next((r for r in top if r["crop"] == targeted), None)

        return {
            "state": state, "district": district, "scope": scope,
            "season": season_keys[0], "season_label": SEASON_LABEL.get(season_keys[0], season_keys[0].title()),
            "land_ha": land_acres * ACRE_TO_HA if land_acres else None,
            "ranked": [self._public(r, land_acres) for r in top],
            "targeted": self._public(target_row, land_acres) if target_row else None,
            "candidates_considered": len(results),
        }

    def _public(self, r, land_acres):
        info = r["info"]
        name = self.profiles.get(r["crop"], {}).get("name") or r["crop"].title()
        if r["crop"] == "rice":
            name = "Rice (Paddy)"
        risk = "Low" if r["score"] >= 0.68 and r["weather"] >= 0.7 else "High" if r["score"] < 0.45 or r["weather"] < 0.4 else "Medium"
        farm_revenue = r["revenue_ha"] * land_acres * ACRE_TO_HA if r["revenue_ha"] and land_acres else None
        return {
            "crop": name, "key": r["crop"],
            "probability": int(max(30, min(97, round(r["score"] * 100)))),
            "risk": risk,
            "season": SEASON_LABEL.get(r["cand"]["season"], r["cand"]["season"].title()),
            "predicted_yield_q_ha": round(r["yield_q_ha"], 1) if r["yield_q_ha"] else None,
            "yield_vs_district_history_pct": round((r["yield_ratio"] - 1) * 100),
            "price": r["price"],
            "expected_revenue_per_ha": round(r["revenue_ha"]) if r["revenue_ha"] else None,
            "expected_farm_revenue": round(farm_revenue) if farm_revenue else None,
            "water_need": info["water"], "cost_tier": info["cost"],
            "temp_range_c": list(info["temp"]),
            "scores": {k: round(r[k], 2) for k in ("local", "weather", "soil", "yield_score", "revenue_score", "npk") if k in r},
            "cautions": r["notes"],
        }

    def describe_models(self):
        m = getattr(self, "metrics", {}) or {}
        y, p = m.get("yield_model", {}), m.get("price_model", {})
        return {
            "yield_model": f"XGBoost, R² {y.get('model', {}).get('r2_log')} on 2017-2020 hold-out, median error {y.get('model', {}).get('median_abs_pct_error')}%",
            "price_model": f"XGBoost 1-month-ahead, median error {p.get('model_median_abs_pct_error')}%",
            "trained_at": m.get("trained_at"),
        }


ml_service = MLService()
