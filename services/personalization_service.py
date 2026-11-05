"""Personalisation: every prediction is built from the farmer's saved profile.

The farmer fills their farm details once at registration (district, soil,
soil test N/P/K/pH, water, irrigation, land size, budget, main crops). After
that, nothing has to be typed again:

  * recommend()        -> crop ranking for this farm using
                          Random Forest (soil + climate, Crop_recommendation.csv)
                          + XGBoost yield model (district history)
                          + live weather (OpenWeatherMap)
                          + TODAY's mandi prices (live API) for the revenue score
  * build_dashboard()  -> "My Farm" page: weather + alerts, recommended crops,
                          this month's tasks for the farmer's own crops, and
                          mandi prices in their own district (live data.gov.in
                          when available, project dataset otherwise).
"""

import asyncio
import time
from datetime import datetime, timezone

import pandas as pd

from fastapi.concurrency import run_in_threadpool

from core.config import settings
from services.agri_data_service import get_agri_data_service
from services.crop_calendar import crop_activities, crop_key, weather_actions
from services.mandi_api_service import mandi_api_service
from services.ml_service import MLService, season_from_month, SEASON_LABEL, SEASON_MAP
from services import live_price_model
from services.price_forecast_service import price_forecast_service
from services.soil_model_service import soil_model_service
from services.weather_service import WeatherUnavailable, weather_service

ml_service = MLService()

# profile field -> recommendation payload field
PROFILE_MAP = {
    "state_name": "stateName", "district_name": "districtName", "taluka_name": "talukaName",
    "soil_type": "soilType", "water_source": "waterSource", "irrigation_type": "irrigationType",
    "farm_size_acres": "landSize", "budget": "budget", "soil_n": "soilN", "soil_p": "soilP",
    "soil_k": "soilK", "soil_ph": "soilPh", "main_crops": "mainCrops",
}
FORECAST_COMMODITY = {"onion": "Onion", "potato": "Potato", "rice": "Rice", "tomato": "Tomato", "wheat": "Wheat"}
SEASON_FROM_WORD = {"rainy": "rainy", "kharif": "rainy", "monsoon": "rainy", "winter": "winter", "rabi": "winter",
                    "summer": "summer", "zaid": "summer"}


def _empty(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "any"}) or value == []


def merge_profile(payload, user):
    """Fill every field the farmer did not type from their saved profile."""
    merged = dict(payload or {})
    if not user:
        return merged
    for field, target in PROFILE_MAP.items():
        if _empty(merged.get(target)) and not _empty(user.get(field)):
            merged[target] = user[field]
    return merged


def profile_payload(user):
    return merge_profile({}, user)


def season_key(payload):
    word = str(payload.get("season") or "").strip().lower()
    return SEASON_FROM_WORD.get(word) or season_from_month()


async def get_weather_safe(district, state):
    try:
        return await weather_service.get_weather(district=district or "", state=state or "")
    except WeatherUnavailable as exc:
        print(f"Weather unavailable: {exc}")
        return None
    except Exception as exc:
        print(f"Weather error: {exc}")
        return None


def warm_up():
    """Load the ML models once at startup so the first request is fast."""
    ml_service.load()
    soil_model_service.load()


# ---------------------------------------------------------------------------
# Crop recommendation
# ---------------------------------------------------------------------------
async def recommend(payload, weather="fetch"):
    state, district = payload.get("stateName") or "", payload.get("districtName") or ""
    if weather == "fetch":
        weather = await get_weather_safe(district, state)
    skey = season_key(payload)

    soil = await run_in_threadpool(soil_model_service.predict, payload, weather, skey)
    soil_probs = soil["all_probabilities"] if soil else None
    offline = get_agri_data_service()

    try:
        ml_payload = {**payload, "season": skey}
        # pass 1: rank with stored prices, just to know which crops are candidates
        first = await run_in_threadpool(ml_service.recommend, ml_payload, weather, 8, soil_probs)
        # live mandi prices for those candidates (parallel, cached), then pass 2 with live prices
        live = await live_crop_prices(state, district, [r["key"] for r in first["ranked"]])
        ml = await run_in_threadpool(ml_service.recommend, ml_payload, weather, 5, soil_probs,
                                     {k: v["price"] for k, v in live.items()})
    except Exception as exc:
        print(f"ML recommendation unavailable, using dataset scoring: {exc}")
        result = await run_in_threadpool(offline.recommend_crops, payload)
        result.update({"mode": "offline_dataset", "weather": weather, "soilModel": soil})
        result["marketTrend"] = one_year_trend(result.get("marketTrend", {}).get("series", []))
        return result

    ranked = ml["ranked"]
    top = ranked[0] if ranked else {"crop": "Wheat", "key": "wheat"}
    top_live = None
    if top["key"] in live:  # full history + 7-day forecast for the winner only
        top_live = (await live_crop_prices(state, district, [top["key"]], with_history=True)).get(top["key"]) \
            or live[top["key"]]
    if top_live:
        analysis_live = top_live["analysis"]
        snap = analysis_live.get("snapshot") or {}
        local = snap.get("district_markets") or snap.get("best_markets") or []
        price_prediction = {
            "crop": analysis_live["commodity"],
            "rangeMin": min((m["min_price"] for m in local), default=None),
            "rangeMax": max((m["max_price"] for m in local), default=None),
            "modalPrice": top_live["price"]["current"], "arrivalQuantity": f"Live, {snap.get('as_of', '')}",
            "marketFeePct": 1.5, "traderCount": snap.get("reporting_markets"), "unit": "Quintal", "grade": "FAQ",
        }
        trend = live_trend(analysis_live)
    else:
        top_commodity = FORECAST_COMMODITY.get(crop_key(top["crop"]) or "", top["crop"].split(" (")[0])
        records = await run_in_threadpool(offline.find_market_records, state, district, "", top_commodity, 8)
        analysis = await run_in_threadpool(offline.summarize_market_records, records,
                                           {"state": state, "district": district, "market": "", "commodity": top_commodity})
        price_prediction = {
            "crop": top_commodity, "rangeMin": analysis["modeled_range_min"], "rangeMax": analysis["modeled_range_max"],
            "modalPrice": analysis["modeled_modal_price"], "arrivalQuantity": "Mandi dataset",
            "marketFeePct": 1.5, "traderCount": len(records), "unit": "Quintal", "grade": "FAQ",
        }
        trend = one_year_trend(analysis["trend_series"])

    place = f"{district.title() or 'your district'}, {state.title() or 'your state'}"
    bullets = [
        f"Ranked for {place} in the {ml['season_label']} season using {ml['candidates_considered']} crops grown "
        f"in your {'district' if ml['scope'] == 'district' else 'state'} (XGBoost yield model on 1997-2020 data).",
    ]
    if soil:
        src = "your soil test" if soil["soil_test_used"] else f"typical values for {payload.get('soilType') or 'loamy'} soil"
        inp = soil["inputs"]
        bullets.append(
            f"Soil model (Random Forest, 99% test accuracy) used N {inp['N']}, P {inp['P']}, K {inp['K']}, pH {inp['ph']} "
            f"from {src}; its best match is {soil['crops'][0]['crop']} ({soil['crops'][0]['probability']}%)."
        )
    if weather:
        o = weather["outlook"]
        bullets.append(f"Live weather for the next {o['days']} days: {o['temp_min']}-{o['temp_max']} C, "
                       f"{o['total_rain_mm']} mm rain, {o['avg_humidity']}% humidity.")
    priced = [r["crop"] for r in ranked if str((r.get("price") or {}).get("source", "")).startswith("Live")]
    if priced:
        bullets.append(f"Revenue uses today's live mandi prices for {', '.join(priced)} "
                       "(Mandi Prices API); other crops use MSP / reference prices.")
    if payload.get("landSize"):
        bullets.append(f"Revenue estimates are for your {payload['landSize']} acres"
                       + (f" with a budget of Rs {int(float(payload['budget'])):,}." if payload.get("budget") else "."))

    return {
        "mode": "ml_live_weather",
        "recommendedCrops": ranked,
        "desiredCropAssessment": ml.get("targeted"),
        "explanation": {"bullets": bullets},
        "pricePrediction": price_prediction,
        "marketTrend": trend,
        "risk": {"level": ranked[0]["risk"] if ranked else "Medium"},
        "weather": weather,
        "soilModel": soil,
        "season": ml["season_label"],
        "models": ml_service.describe_models(),
    }


# ---------------------------------------------------------------------------
# Live prices and trends
# ---------------------------------------------------------------------------
LIVE_SKIP = {"sugarcane", "dry chillies", "cotton lint"}  # live rows are a different product / unit


async def live_crop_prices(state, district, crop_keys, with_history=False):
    """{ml crop key: {"price": price dict for ml_service, "analysis": live model result}}
    Candidates only need today's price (one API call each); history + forecast is fetched
    later for the top crop only, to stay inside the API's rate limit."""
    keys = [k for k in dict.fromkeys(crop_keys) if k and k not in LIVE_SKIP]
    results = await asyncio.gather(*(live_price_model.analyze(state, k, district, with_history=with_history)
                                     for k in keys), return_exceptions=True)
    out = {}
    for key, res in zip(keys, results):
        if not isinstance(res, dict):
            continue
        now = live_price_model.latest_price(res, district)
        if not now:
            continue
        fc = res.get("forecast") or {}
        out[key] = {"analysis": res, "price": {
            "price": fc.get("end_price") or now["price"], "current": now["price"],
            "change_pct": fc.get("change_pct", 0.0), "as_of": now["as_of"], "scope": now["scope"].title(),
            "source": f"Live mandi price ({now['scope']}, {now['as_of']})",
        }}
    return out


def live_trend(analysis):
    series = analysis.get("series") or []
    where = analysis.get("market") or f"{analysis.get('state')} average"
    days = len((analysis.get("forecast") or {}).get("points", []))
    note = (f"Live daily {analysis['commodity']} prices at {where}, {series[0]['label']} - {series[-1]['label']}"
            + (f", plus the next {days} market days (f = forecast)." if days else ".")) if series else "Live prices"
    return {"series": live_price_model.chart_series(analysis, history_days=30), "note": note, "live": True}


def one_year_trend(series):
    """Keep only the last 12 months. If nothing is that recent, still show the old data with a note."""
    dated = []
    for point in series or []:
        d = pd.to_datetime(str(point.get("label", "")), errors="coerce", format="mixed")
        dated.append((d, point))
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    recent = [p for d, p in dated if not pd.isna(d) and d >= cutoff]
    if recent:
        return {"series": recent, "note": "Dataset prices from the last 12 months.", "live": False}
    known = [d for d, _ in dated if not pd.isna(d)]
    latest = max(known).strftime("%b %Y") if known else "an older year"
    return {"series": list(series or []), "live": False,
            "note": f"No prices from the last 12 months for this crop - showing the latest available dataset prices (up to {latest})."}


# ---------------------------------------------------------------------------
# Mandi prices for the farmer's district
# ---------------------------------------------------------------------------
async def district_mandi(state, district, crops):
    """Mandi prices for the farmer: live API first, project dataset if it is down."""
    offline = get_agri_data_service()
    live, source = await mandi_api_service.get_farmer_prices(state, district, crops)
    if live:
        records = offline.normalize_market_records(live)
        for record, raw in zip(records, live):
            record["is_my_crop"] = raw.get("is_my_crop", False)
            record["in_my_district"] = raw.get("in_my_district", False)
    else:
        records = []
        for crop in crops:
            name = FORECAST_COMMODITY.get(crop_key(crop) or "", str(crop).split(" (")[0])
            records += await run_in_threadpool(offline.find_market_records, state, district, "", name, 6)
        records = offline.normalize_market_records(records)
        my_keys = {crop_key(c) for c in crops if crop_key(c)}
        for record in records:
            record["is_my_crop"] = crop_key(record["commodity"]) in my_keys
            record["in_my_district"] = True
        source = "Project mandi dataset (live APIs unavailable)"

    unique = {}
    for record in records:
        unique[(record["market"], record["commodity"], record["variety"], record["arrival_date"], record["modal_price"])] = record
    records = sorted(unique.values(), key=lambda r: (not r["is_my_crop"], not r["in_my_district"], r["commodity"], -r["modal_price"]))
    per_commodity, capped = {}, []
    for record in records:  # at most 6 mandis per commodity so every crop gets space
        per_commodity[record["commodity"]] = per_commodity.get(record["commodity"], 0) + 1
        if per_commodity[record["commodity"]] <= 6:
            capped.append(record)
    records = capped
    parsed = pd.to_datetime(pd.Series([r["arrival_date"] for r in records], dtype="object"),
                            errors="coerce", format="mixed", dayfirst=False)
    latest = parsed.max() if len(parsed) else None
    return {
        "source": source,
        "is_live": bool(live),
        "as_of": latest.strftime("%d %b %Y") if latest is not None and not pd.isna(latest) else None,
        "records": records[:80],
        "commodities": len({r["commodity"] for r in records}),
        "api_error": mandi_api_service.last_error if not live else None,
    }


async def price_outlooks(state, district, crops):
    """Live 7-day forecast for each of the farmer's crops; old monthly dataset model as fallback."""
    live = await live_price_model.analyze_many(state, crops, district)
    outlooks = [{"crop": r["commodity"], "live": True, **r, "chart": live_price_model.chart_series(r)} for r in live
                if r.get("forecast") or r.get("snapshot")]
    have = " ".join(o["crop"].lower() for o in outlooks)
    for crop in crops:
        commodity = FORECAST_COMMODITY.get(crop_key(crop) or "")
        if not commodity or commodity.lower() in have:
            continue
        forecast = await run_in_threadpool(price_forecast_service.forecast, state, district, "", commodity)
        if forecast:
            outlooks.append({"crop": commodity, "live": False, "monthly": forecast})
    return outlooks


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
_dashboard_cache = {}


async def build_dashboard(user, force=False):
    cache_key = (user["id"], str(user.get("updated_at")))
    cached = _dashboard_cache.get(cache_key)
    if cached and not force and time.time() - cached[0] < settings.DASHBOARD_CACHE_SECONDS:
        return {**cached[1], "cached": True}

    payload = profile_payload(user)
    state, district = payload.get("stateName", ""), payload.get("districtName", "")
    my_crops = [c for c in (payload.get("mainCrops") or []) if str(c).strip()]

    # weather and mandi prices are independent network calls - fetch them together
    weather, mandi = await asyncio.gather(
        get_weather_safe(district, state),
        district_mandi(state, district, my_crops),
    )
    recommendation = await recommend(payload, weather=weather)
    crops_for_tasks = my_crops or [c["crop"] for c in recommendation.get("recommendedCrops", [])[:3]]
    month = datetime.now().month
    activities = [a for a in (crop_activities(c, month) for c in crops_for_tasks) if a]
    outlooks = await price_outlooks(state, district, my_crops or crops_for_tasks)

    skey = season_from_month(month)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "farmer": {
            "name": user.get("full_name") or "Farmer",
            "state": state, "district": district, "taluka": payload.get("talukaName"),
            "soil_type": payload.get("soilType"), "land_acres": payload.get("landSize"),
            "water_source": payload.get("waterSource"), "irrigation": payload.get("irrigationType"),
            "budget": payload.get("budget"), "main_crops": my_crops,
            "soil_test": {k: payload.get(k) for k in ("soilN", "soilP", "soilK", "soilPh")},
        },
        "season": {"key": skey, "label": SEASON_LABEL.get(SEASON_MAP[skey][0], skey.title()),
                   "month": datetime.now().strftime("%B")},
        "weather": weather,
        "weather_actions": weather_actions(weather),
        "recommendation": recommendation,
        "activities": activities,
        "mandi": mandi,
        "price_outlooks": outlooks,
        "sources": {
            "weather": "OpenWeatherMap (live)" if weather else "unavailable - set OPENWEATHER_API_KEY",
            "mandi": mandi["source"],
            "models": "Random Forest soil model + XGBoost yield/price models",
        },
    }
    _dashboard_cache[cache_key] = (time.time(), data)
    return data


def farmer_context(user):
    """One line describing the farmer, given to the chatbot so answers are local."""
    if not user:
        return ""
    p = profile_payload(user)
    parts = [f"Farmer location: {p.get('districtName', '')}, {p.get('stateName', '')}"]
    if p.get("talukaName"):
        parts.append(f"taluka {p['talukaName']}")
    if p.get("soilType"):
        parts.append(f"soil {p['soilType']}")
    if p.get("landSize"):
        parts.append(f"{p['landSize']} acres")
    if p.get("waterSource"):
        parts.append(f"water: {p['waterSource']}")
    if p.get("irrigationType"):
        parts.append(f"irrigation: {p['irrigationType']}")
    if p.get("mainCrops"):
        parts.append("grows " + ", ".join(p["mainCrops"]))
    parts.append(f"current season: {SEASON_LABEL.get(SEASON_MAP[season_from_month()][0])}, {datetime.now():%B %Y}")
    return "; ".join(parts)
