import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from core.security import get_current_user, get_optional_user
from services import personalization_service as personal
from services.crop_calendar import crop_activities, weather_actions

router = APIRouter(prefix="/api", tags=["dashboard"])


def _require_location(user):
    if not user.get("state_name") or not user.get("district_name"):
        raise HTTPException(status_code=400, detail="Please complete your farm profile (state and district).")


@router.get("/dashboard")
async def my_dashboard(refresh: bool = Query(default=False), user=Depends(get_current_user)):
    """Everything for the logged-in farmer's own farm, built from their saved profile."""
    _require_location(user)
    return await personal.build_dashboard(user, force=refresh)


@router.get("/my-weather")
async def my_weather(user=Depends(get_current_user)):
    _require_location(user)
    weather = await personal.get_weather_safe(user["district_name"], user["state_name"])
    if not weather:
        raise HTTPException(status_code=503, detail="Live weather is unavailable right now.")
    return {**weather, "actions": weather_actions(weather)}


@router.get("/my-mandi")
async def my_mandi(user=Depends(get_current_user)):
    """Today's mandi prices in the farmer's district, their own crops first."""
    _require_location(user)
    return await personal.district_mandi(user["state_name"], user["district_name"], user.get("main_crops") or [])


@router.get("/crop-calendar")
async def crop_calendar(crop: str, month: int = Query(default=0, ge=0, le=12)):
    activity = crop_activities(crop, month or None)
    if not activity:
        raise HTTPException(status_code=404, detail=f"No calendar for '{crop}' yet.")
    return activity


# ---------------------------------------------------------------- live mandi
@router.get("/mandi/options")
async def mandi_options(state: str = Query(default="Maharashtra")):
    """Live dropdown data: supported states, districts and mandis of a state, and traded commodities."""
    from services.mandi_api_service import COMMODITY_NAMES, mandi_api_service

    async def safe(coro, default):
        try:
            return await coro
        except Exception as exc:
            print(f"Mandi options: {exc}")
            return default

    states, markets, latest, listed = await asyncio.gather(
        safe(mandi_api_service.get_states(), []),
        safe(mandi_api_service.get_markets(state), []),
        safe(mandi_api_service.fetch_live(state), []),
        safe(mandi_api_service._get_json("commodities", {"state": state}, 6 * 3600), []),
    )
    by_district = {}
    for m in markets:
        by_district.setdefault(m["district"] or "Other", []).append(m["market"])
    commodities = sorted({*(r.get("commodity") for r in latest if r.get("commodity")), *listed,
                          *COMMODITY_NAMES.values()} - {None, ""})
    return {
        "state": state,
        "states": states,
        "districts": sorted(by_district),
        "markets_by_district": {d: sorted(set(v)) for d, v in sorted(by_district.items())},
        "commodities": commodities,
        "live": bool(markets),
    }


@router.get("/mandi/analysis")
async def mandi_analysis(commodity: str, state: str = "", district: str = "", market: str = "",
                         user=Depends(get_optional_user)):
    """Live price analysis for one commodity (snapshot, daily history, ensemble forecast, advice)."""
    from services import live_price_model

    state = state or (user or {}).get("state_name") or "Maharashtra"
    district = district or (user or {}).get("district_name") or ""
    result = await live_price_model.analyze(state, commodity, district, market)
    if not result:
        raise HTTPException(status_code=503, detail="Live mandi prices are unavailable right now.")
    return {**result, "chart": live_price_model.chart_series(result)}
