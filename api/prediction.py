import json
from fastapi import APIRouter, Depends, HTTPException
from core.security import get_optional_user
from models.request_models import PricePredictionRequest
from core.config import settings
from services.agri_data_service import get_agri_data_service
import asyncio

from services import live_price_model
from services.mandi_api_service import mandi_api_service, same_district
from services.personalization_service import one_year_trend
from services.rag_service import get_rag_service

router = APIRouter(prefix="/api")


def as_payload(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def get_ready_rag_service():
    if not settings.RAG_ENABLED:
        return None

    rag = get_rag_service()
    return rag if rag.available else None

@router.post("/market-prediction")
async def predict_crop_price(req: PricePredictionRequest, user=Depends(get_optional_user)):
    """
    0. Fills state / district / commodity from the farmer's profile when left empty.
    1. Normalizes input with AI.
    2. Fetches live records from Mandi API.
    3. Analyzes results with AI.
    """
    if user:
        req.state = req.state or user.get("state_name") or ""
        req.district = req.district or user.get("district_name") or ""
        req.commodity = req.commodity or next(iter(user.get("main_crops") or []), "")
    if not req.state or not req.commodity:
        raise HTTPException(status_code=400, detail="Please choose a state and commodity.")
    offline_service = get_agri_data_service()
    request_payload = as_payload(req)
    normal = offline_service.normalize_market_input(request_payload)

    # 1. AI-Driven Input Normalization
    rag = get_ready_rag_service()
    if rag is not None:
        try:
            rag_normal = await asyncio.wait_for(
                rag.normalize_market_params(request_payload),
                timeout=4,
            )
            if isinstance(rag_normal, dict):
                normal.update({key: value for key, value in rag_normal.items() if value})
        except Exception as e:
            print(f"AI normalization unavailable, using dataset input: {e}")

    # 2. Prefer the local dataset for a fast response, then try live sources
    #    when local records are unavailable (or local-first mode is disabled).
    local_records = offline_service.find_market_records(
        state=normal.get("state", req.state),
        district=normal.get("district", req.district),
        market=normal.get("market", req.market),
        commodity=normal.get("commodity", req.commodity),
    ) if settings.FAST_MARKET_LOCAL_FIRST else []
    records = local_records
    is_live = False
    if not records:
        try:
            records = await asyncio.wait_for(
                mandi_api_service.get_market_prices(
                    state=normal.get("state", req.state),
                    district=normal.get("district", req.district),
                    market=normal.get("market", req.market),
                    commodity=normal.get("commodity", req.commodity),
                ),
                timeout=6,
            )
        except Exception as e:
            print(f"Live mandi lookup failed: {e}")
            records = []
        is_live = bool(records)
    if not records and not settings.FAST_MARKET_LOCAL_FIRST:
        local_records = offline_service.find_market_records(
            state=normal.get("state", req.state),
            district=normal.get("district", req.district),
            market=normal.get("market", req.market),
            commodity=normal.get("commodity", req.commodity),
        )
    records = records or local_records

    if not isinstance(records, list):
        records = []

    records = offline_service.normalize_market_records(records)
    if not records and local_records:
        records = offline_service.normalize_market_records(local_records)

    # 3. Live price model: daily history + ensemble forecast + best mandis to sell
    live = None
    if is_live:
        try:
            live = await asyncio.wait_for(
                live_price_model.analyze(
                    normal.get("state", req.state), normal.get("commodity", req.commodity),
                    normal.get("district", req.district), normal.get("market", req.market)),
                timeout=6,
            )
        except Exception as e:
            print(f"Live price model failed: {e}")

    analysis_data = offline_service.summarize_market_records(records, normal)
    if live and (live.get("forecast") or live.get("snapshot")):
        fc, snap = live.get("forecast"), live.get("snapshot")
        change = (fc or {}).get("change_pct", 0)
        analysis_data.update({
            "sentiment": "Bullish" if change >= 3 else "Bearish" if change <= -3 else "Stable",
            "prediction_insight": live["advice"][0] if live["advice"] else analysis_data.get("prediction_insight"),
            "analysis_bullets": live["advice"][1:] + ([
                f"Forecast model: {fc['method']}. Tested on the last {fc['backtest_days']} market days: "
                f"error {fc['backtest_error_pct']}% (no-change baseline {fc['naive_error_pct']}%)."] if fc else []),
            "trend_series": live_price_model.chart_series(live, history_days=30),
            "trend_note": f"Live daily prices ({live['market'] or live['state'] + ' average'}) and the next "
                          f"{len((fc or {}).get('points', []))} market days (f = forecast).",
            "modeled_modal_price": (fc or {}).get("next_price") or (snap or {}).get("state_median"),
        })
    else:
        trend = one_year_trend(analysis_data.get("trend_series"))
        analysis_data["trend_series"], analysis_data["trend_note"] = trend["series"], trend["note"]

    # 4. AI narrative (optional)
    if rag is not None:
        try:
            raw_analysis = await asyncio.wait_for(
                rag.analyze_live_market_data(records, normal),
                timeout=5,
            )
            clean_json = raw_analysis.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            ai_analysis = json.loads(clean_json)
            # Keep the statistical forecast even when the LLM writes the narrative.
            for key in ("forecast", "trend_series", "modeled_modal_price"):
                if analysis_data.get(key) is not None:
                    ai_analysis[key] = analysis_data[key]
            analysis_data = ai_analysis
        except Exception as e:
            print(f"AI market analysis unavailable, using dataset analysis: {e}")

    return {
        "records": records,
        "analysis": analysis_data,
        "normalized_input": normal,
        "source": "live" if is_live else "dataset",
        "live_model": live,
    }


async def live_heatmap(state, district):
    """What each mandi in the district is trading right now (one API call per mandi, cached)."""
    markets = [m for m in await mandi_api_service.get_markets(state) if same_district(m["district"], district)]
    if not markets:
        return []
    results = await asyncio.gather(*(mandi_api_service.fetch_live(state, "", m["market"]) for m in markets[:12]),
                                   return_exceptions=True)
    cells = {}
    for market, rows in zip(markets, results):
        if not isinstance(rows, list):
            continue
        for row in rows:
            key = (market["market"], row.get("commodity"))
            cell = cells.setdefault(key, {"subRegion": market["market"], "cropName": row.get("commodity"),
                                          "intensity": 0, "prices": []})
            cell["intensity"] += 1
            cell["prices"].append(float(row.get("modal_price") or 0))
    data = []
    for cell in cells.values():
        prices = [p for p in cell.pop("prices") if p > 0]
        cell["avgPrice"] = int(sum(prices) / len(prices)) if prices else None
        data.append(cell)
    return sorted(data, key=lambda c: (-c["intensity"], c["subRegion"]))

@router.post("/crop-intensity-heatmap")
async def get_crop_heatmap(req: PricePredictionRequest, user=Depends(get_optional_user)):
    """
    Retrieves data for HeatMap visualization.
    Aggregates crops by Mandi (Sub-region) for the given district.
    """
    if user:
        req.state = req.state or user.get("state_name") or ""
        req.district = req.district or user.get("district_name") or ""
    offline_service = get_agri_data_service()
    try:
        live_cells = await live_heatmap(req.state, req.district)
    except Exception as e:
        print(f"Live heatmap unavailable: {e}")
        live_cells = []
    if live_cells:
        return {"state": req.state, "district": req.district, "data": live_cells,
                "source": "Live - Mandi Prices API (records per mandi and commodity)"}
    rag = get_ready_rag_service()
    if rag is None:
        return {
            "state": req.state,
            "district": req.district,
            "data": offline_service.build_heatmap(req.state, req.district),
        }

    try:
        # Search for all mentions of crops in the district to build a distribution map
        query = f"{req.district} {req.state}"
        
        # Get from both collections for a richer map
        docs = rag.get_context(query, "MandiPrices", k=30)
        docs += rag.get_context(query, "CropProduction", k=30)
        
        import re
        distribution = {}
        for doc in docs:
            content = doc.page_content
            # Try to get Mandi Name
            mandi = doc.metadata.get("Market") or doc.metadata.get("Mandi")
            if not mandi:
                match = re.search(r'(?:Mandi|Market|District)[:\s]+([a-zA-Z0-9\s]+)', content, re.IGNORECASE)
                mandi = match.group(1).strip() if match else f"Agro Market ({req.district})"
            
            # Try to get Crop/Commodity
            commodity = doc.metadata.get("Commodity") or doc.metadata.get("Crop")
            if not commodity:
                match = re.search(r'(?:Commodity|Crop)[:\s]+([a-zA-Z0-9\s]+)', content, re.IGNORECASE)
                commodity = match.group(1).strip() if match else "Various"

            # Clean and categorize
            mandi = mandi.split(',')[0].strip()
            
            if mandi not in distribution:
                distribution[mandi] = {}
            if commodity not in distribution[mandi]:
                distribution[mandi][commodity] = 0
            distribution[mandi][commodity] += 1

        # Format for frontend (HeatMap often expects [{region, crop, value}, ...])
        formatted_data = []
        for mandi_name, crops in distribution.items():
            for crop_name, count in crops.items():
                formatted_data.append({
                    "subRegion": mandi_name,
                    "cropName": crop_name,
                    "intensity": count
                })
        
        # Simulated fallback if dataset is sparse for this specific state/district
        if len(formatted_data) == 0:
            formatted_data = await rag.generate_dynamic_heatmap(req.state, req.district)
        
        return {
            "state": req.state,
            "district": req.district,
            "data": formatted_data
        }
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        return {
            "state": req.state,
            "district": req.district,
            "data": offline_service.build_heatmap(req.state, req.district),
        }
