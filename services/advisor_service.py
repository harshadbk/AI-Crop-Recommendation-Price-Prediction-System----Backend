"""Glue: live weather + ML models + LLM -> crop recommendation and chat."""

from datetime import datetime

from fastapi.concurrency import run_in_threadpool

from services.agri_data_service import get_agri_data_service
from services.llm_service import llm_service
from services.ml_service import ml_service, season_from_month
from services.weather_service import WeatherUnavailable, weather_service


async def get_weather_safe(district, state):
    try:
        return await weather_service.get_weather(district=district or "", state=state or "")
    except WeatherUnavailable as exc:
        print(f"Weather unavailable: {exc}")
        return None


def _fmt_rs(value):
    return f"Rs {value:,.0f}" if value else "n/a"


def rule_bullets(ml, weather, payload):
    top = ml["ranked"][0]
    where = (ml["district"] or ml["state"] or "your area").title()
    bullets = [
        f"{top['crop']} ranks first for {ml['season_label']} in {where}: local history score "
        f"{top['scores']['local']}, weather fit {top['scores']['weather']}, soil fit {top['scores']['soil']}.",
    ]
    if top.get("predicted_yield_q_ha"):
        bullets.append(
            f"The XGBoost yield model expects about {top['predicted_yield_q_ha']} q/ha "
            f"({top['yield_vs_district_history_pct']:+d}% vs this district's history)."
        )
    if top.get("price"):
        p = top["price"]
        bullets.append(f"Price outlook {_fmt_rs(p['price'])}/quintal ({p['source']}, {p['scope']}, as of {p['as_of']}).")
    if top.get("expected_farm_revenue"):
        bullets.append(f"Estimated gross revenue for {payload.get('landSize')} acres: {_fmt_rs(top['expected_farm_revenue'])}.")
    if weather:
        o = weather["outlook"]
        bullets.append(
            f"Live weather: {weather['current']['temp']} C now, {o['temp_min']}-{o['temp_max']} C and "
            f"{o['total_rain_mm']} mm rain over the next {o['days']} days."
        )
        bullets.extend(weather.get("alerts", [])[:2])
    if top.get("cautions"):
        bullets.append("Watch out: " + "; ".join(top["cautions"]) + ".")
    return bullets


async def recommend(payload):
    """Response in the shape the frontend Result page expects, plus ML/weather detail."""
    weather = await get_weather_safe(payload.get("districtName"), payload.get("stateName"))
    ml = await run_in_threadpool(ml_service.recommend, payload, weather)
    top = ml["ranked"][0]

    bullets, explained_by = rule_bullets(ml, weather, payload), "rules"
    if llm_service.enabled:
        try:
            facts = {
                "farm_inputs": {**payload, "landSize_unit": "acres", "budget_unit": "INR total"},
                "note": "expected_farm_revenue is for the whole farm; expected_revenue_per_ha is per hectare.",
                "weather": weather and {k: weather[k] for k in ("current", "outlook", "alerts")},
                "ranking": ml["ranked"], "targeted_crop": ml["targeted"],
            }
            bullets, explained_by = await llm_service.explain(facts), "llm"
        except Exception as exc:
            print(f"LLM explanation failed, using rule-based bullets: {exc}")

    # Mandi detail for the top crop from the existing dataset service.
    agri = get_agri_data_service()
    market_name = "paddy" if top["key"] == "rice" else top["key"]
    records = await run_in_threadpool(
        agri.find_market_records, payload.get("stateName"), payload.get("districtName"), "", market_name, 8
    )
    analysis = agri.summarize_market_records(records, {
        "state": payload.get("stateName"), "district": payload.get("districtName"),
        "market": payload.get("marketName") or "", "commodity": top["crop"],
    })
    price = top.get("price") or {}
    modal = price.get("price") or analysis["modeled_modal_price"]

    return {
        "mode": "ml_live_weather",
        "recommendedCrops": ml["ranked"],
        "desiredCropAssessment": ml["targeted"],
        "explanation": {"bullets": bullets, "source": explained_by},
        "pricePrediction": {
            "crop": top["crop"],
            "modalPrice": modal,
            "rangeMin": round(modal * 0.9),
            "rangeMax": round(modal * 1.1),
            "source": price.get("source", "mandi dataset"),
            "changePct": price.get("change_pct"),
            "asOf": price.get("as_of"),
            "arrivalQuantity": "Model estimate",
            "marketFeePct": 1.5, "traderCount": len(records) or 12,
            "unit": "Quintal", "grade": "FAQ",
        },
        "marketTrend": {"series": analysis["trend_series"]},
        "risk": {"level": top["risk"]},
        "weather": weather,
        "ml": {
            "season": ml["season_label"], "scope": ml["scope"],
            "candidates": ml["candidates_considered"], "models": ml_service.describe_models(),
        },
    }


async def build_chat_context(message, history, user=None):
    """Real-time facts for the LLM: location -> live weather, ML ranking, crop price."""
    text_pool = [message] + [
        h.get("content") or "" for h in reversed(history or []) if h.get("role") == "user"
    ][:4]
    loaded = await run_in_threadpool(ml_service.load)

    state = district = None
    if loaded:
        for text in text_pool:
            state, district = ml_service.find_location_in_text(text)
            if state:
                break
    if not state and user:
        state, district = user.get("state_name") or None, user.get("district_name") or None

    crop = None
    if loaded:
        for text in text_pool:
            crop = ml_service.find_crop_in_text(text)
            if crop:
                break

    lines = [f"Today: {datetime.now():%d %b %Y}. Current Indian season: {season_from_month()}."]
    if user:
        profile = ", ".join(
            f"{k}={user.get(k)}" for k in ("state_name", "district_name", "soil_type", "water_source",
                                           "irrigation_type", "farm_size_acres", "main_crops") if user.get(k)
        )
        if profile:
            lines.append("Farmer profile: " + profile)
    if not state:
        lines.append("Location unknown: ask the farmer for their district if location matters.")
        return "\n".join(lines)

    weather = await get_weather_safe(district or "", state)
    if weather:
        c, o = weather["current"], weather["outlook"]
        lines.append(
            f"LIVE WEATHER {weather['location']['district']}, {weather['location']['state']} (OpenWeather): "
            f"now {c['temp']} C, {c['condition']}, humidity {c['humidity']}%, wind {c['wind_ms']} m/s. "
            f"Next {o['days']} days: {o['temp_min']}-{o['temp_max']} C, rain {o['total_rain_mm']} mm on "
            f"{o['rainy_days']} days, avg humidity {o['avg_humidity']}%."
        )
        for d in weather["daily"]:
            lines.append(f"  {d['label']}: {d['temp_min']}-{d['temp_max']} C, rain {d['rain_mm']} mm "
                         f"({d['rain_chance_pct']}%), {d['condition']}")
        if weather["alerts"]:
            lines.append("Weather alerts: " + " | ".join(weather["alerts"]))
        if weather["good_spray_days"]:
            lines.append("Good spraying days: " + ", ".join(weather["good_spray_days"]))

    if loaded:
        u = user or {}
        payload = {"stateName": state, "districtName": district or "", "targetedCrop": crop or "",
                   "soilType": u.get("soil_type") or "", "waterSource": u.get("water_source") or "",
                   "irrigationType": u.get("irrigation_type") or ""}
        try:
            ml = await run_in_threadpool(ml_service.recommend, payload, weather)
            lines.append(f"ML CROP RANKING for {(district or state).title()} "
                         f"({ml['season_label']}, {ml['scope']}-level data):")
            for i, r in enumerate(ml["ranked"], 1):
                p = r.get("price") or {}
                caution = f"; cautions: {'; '.join(r['cautions'])}" if r["cautions"] else ""
                lines.append(
                    f"  {i}. {r['crop']}: score {r['probability']}/100, risk {r['risk']}, yield ~{r['predicted_yield_q_ha']} q/ha, "
                    f"price {_fmt_rs(p.get('price'))}/q ({p.get('source', 'n/a')}), revenue/ha "
                    f"{_fmt_rs(r['expected_revenue_per_ha'])}{caution}"
                )
            t = ml.get("targeted")
            if t and all(t["key"] != r["key"] for r in ml["ranked"]):
                lines.append(f"  Asked-about crop {t['crop']}: score {t['probability']}/100, risk {t['risk']}, "
                             f"cautions: {'; '.join(t['cautions']) or 'none'}")
        except Exception as exc:
            lines.append(f"ML ranking unavailable: {exc}")

        if crop:
            s_key, d_key = ml_service.resolve_location(state, district)
            p = await run_in_threadpool(ml_service.price_outlook, crop, s_key, d_key)
            if p:
                lines.append(
                    f"PRICE {crop}: current {_fmt_rs(p['current'])}/q, next-month forecast {_fmt_rs(p['price'])}/q "
                    f"({p['change_pct']:+}%), {p['source']}, {p['scope']}, data as of {p['as_of']}."
                )
    return "\n".join(lines)


async def chat(message, history, user=None):
    context = await build_chat_context(message, history, user)
    reply = await llm_service.chat(message, history, context)
    reply["mode"] = "llm_live_context"
    return reply
