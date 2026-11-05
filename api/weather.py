from fastapi import APIRouter, HTTPException, Query

from services.weather_service import WeatherUnavailable, weather_service

router = APIRouter(prefix="/api", tags=["weather"])


@router.get("/weather")
async def get_weather(district: str = Query(default=""), state: str = Query(default="")):
    """Live current weather, 5-day outlook and farm alerts for a district."""
    try:
        return await weather_service.get_weather(district=district, state=state)
    except WeatherUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
