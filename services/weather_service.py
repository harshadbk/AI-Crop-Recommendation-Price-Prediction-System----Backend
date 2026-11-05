"""Live weather from OpenWeatherMap (current conditions + 5-day / 3-hour forecast).

Flow: district/state -> geocode (cached forever) -> current + forecast (cached
WEATHER_CACHE_SECONDS) -> farmer-friendly summary with alerts.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from core.config import settings

BASE = "https://api.openweathermap.org"


class WeatherUnavailable(RuntimeError):
    pass


class WeatherService:
    def __init__(self):
        self._geo_cache = {}
        self._cache = {}
        self._locks = defaultdict(asyncio.Lock)

    @property
    def enabled(self):
        return bool(settings.OPENWEATHER_API_KEY)

    async def _get(self, client, path, **params):
        params["appid"] = settings.OPENWEATHER_API_KEY
        response = await client.get(f"{BASE}{path}", params=params)
        if response.status_code == 401:
            raise WeatherUnavailable("OpenWeather API key is invalid.")
        response.raise_for_status()
        return response.json()

    async def _geocode(self, client, district, state):
        cache_key = (district.lower(), state.lower())
        if cache_key in self._geo_cache:
            return self._geo_cache[cache_key]
        queries = [q for q in (f"{district},{state},IN", f"{district},IN", f"{state},IN") if not q.startswith(",")]
        for query in queries:
            results = await self._get(client, "/geo/1.0/direct", q=query, limit=1)
            if results:
                place = results[0]
                found = {
                    "lat": place["lat"], "lon": place["lon"],
                    "name": place.get("name") or district or state,
                    "state": place.get("state") or state,
                    "matched_query": query,
                }
                self._geo_cache[cache_key] = found
                return found
        raise WeatherUnavailable(f"Could not find location '{district}, {state}'.")

    async def get_weather(self, district="", state=""):
        """Weather summary for a district. Raises WeatherUnavailable on failure."""
        if not self.enabled:
            raise WeatherUnavailable("OPENWEATHER_API_KEY is not set.")
        district, state = (district or "").strip(), (state or "").strip()
        if not district and not state:
            raise WeatherUnavailable("A district or state is required for weather.")

        cache_key = (district.lower(), state.lower())
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < settings.WEATHER_CACHE_SECONDS:
            return cached[1]

        async with self._locks[cache_key]:
            cached = self._cache.get(cache_key)
            if cached and time.time() - cached[0] < settings.WEATHER_CACHE_SECONDS:
                return cached[1]
            try:
                async with httpx.AsyncClient(timeout=8) as client:
                    place = await self._geocode(client, district, state)
                    current, forecast = await asyncio.gather(
                        self._get(client, "/data/2.5/weather", lat=place["lat"], lon=place["lon"], units="metric"),
                        self._get(client, "/data/2.5/forecast", lat=place["lat"], lon=place["lon"], units="metric"),
                    )
            except WeatherUnavailable:
                raise
            except Exception as exc:
                if cached:  # serve stale data rather than nothing
                    return {**cached[1], "stale": True}
                raise WeatherUnavailable(f"Weather service unreachable: {exc}") from exc

            summary = self._summarize(place, current, forecast)
            self._cache[cache_key] = (time.time(), summary)
            return summary

    @staticmethod
    def _summarize(place, current, forecast):
        tz_shift = forecast.get("city", {}).get("timezone", 19800)
        days = {}
        for item in forecast.get("list", []):
            local = datetime.fromtimestamp(item["dt"] + tz_shift, tz=timezone.utc)
            day = days.setdefault(local.strftime("%Y-%m-%d"), {
                "date": local.strftime("%Y-%m-%d"), "label": local.strftime("%a %d %b"),
                "temp_min": 99.0, "temp_max": -99.0, "humidity": [], "rain_mm": 0.0,
                "wind_max": 0.0, "pop_max": 0.0, "conditions": defaultdict(int),
            })
            main = item.get("main", {})
            day["temp_min"] = min(day["temp_min"], main.get("temp_min", main.get("temp", 0)))
            day["temp_max"] = max(day["temp_max"], main.get("temp_max", main.get("temp", 0)))
            day["humidity"].append(main.get("humidity", 0))
            day["rain_mm"] += (item.get("rain") or {}).get("3h", 0.0)
            day["wind_max"] = max(day["wind_max"], (item.get("wind") or {}).get("speed", 0.0))
            day["pop_max"] = max(day["pop_max"], item.get("pop", 0.0))
            for w in item.get("weather", []):
                day["conditions"][w.get("description", "")] += 1

        daily = []
        for day in list(days.values())[:6]:
            daily.append({
                "date": day["date"], "label": day["label"],
                "temp_min": round(day["temp_min"], 1), "temp_max": round(day["temp_max"], 1),
                "humidity": round(sum(day["humidity"]) / max(len(day["humidity"]), 1)),
                "rain_mm": round(day["rain_mm"], 1),
                "rain_chance_pct": round(day["pop_max"] * 100),
                "wind_max_ms": round(day["wind_max"], 1),
                "condition": max(day["conditions"], key=day["conditions"].get) if day["conditions"] else "",
            })

        temps_max = [d["temp_max"] for d in daily] or [current["main"]["temp"]]
        temps_min = [d["temp_min"] for d in daily] or [current["main"]["temp"]]
        total_rain = round(sum(d["rain_mm"] for d in daily), 1)
        avg_humidity = round(sum(d["humidity"] for d in daily) / max(len(daily), 1)) if daily else current["main"]["humidity"]

        alerts = []
        if max(temps_max) >= 40:
            alerts.append(f"Heat stress: up to {max(temps_max):.0f}°C. Irrigate in the evening and mulch soil.")
        if min(temps_min) <= 4:
            alerts.append(f"Frost risk: lows near {min(temps_min):.0f}°C. Protect nurseries and young plants.")
        heavy = [d for d in daily if d["rain_mm"] >= 50]
        if heavy:
            alerts.append(f"Heavy rain expected on {heavy[0]['label']} ({heavy[0]['rain_mm']} mm). Ensure field drainage; delay spraying.")
        if any(d["wind_max_ms"] >= 12 for d in daily):
            alerts.append("Strong winds expected. Avoid pesticide spraying and stake tall crops.")
        if avg_humidity >= 85 and max(temps_max) >= 25:
            alerts.append("Warm and very humid: higher risk of fungal disease. Scout crops for blight and mildew.")
        if total_rain < 1 and max(temps_max) >= 33:
            alerts.append("Dry and hot week ahead: plan irrigation; avoid sowing without assured water.")

        spray_days = [d["label"] for d in daily if d["rain_chance_pct"] < 30 and d["wind_max_ms"] < 6]

        return {
            "location": {"district": place["name"], "state": place["state"], "lat": place["lat"], "lon": place["lon"]},
            "current": {
                "temp": round(current["main"]["temp"], 1),
                "feels_like": round(current["main"].get("feels_like", current["main"]["temp"]), 1),
                "humidity": current["main"]["humidity"],
                "pressure": current["main"].get("pressure"),
                "wind_ms": round((current.get("wind") or {}).get("speed", 0.0), 1),
                "clouds_pct": (current.get("clouds") or {}).get("all"),
                "condition": (current.get("weather") or [{}])[0].get("description", ""),
                "icon": (current.get("weather") or [{}])[0].get("icon", ""),
                "rain_1h_mm": (current.get("rain") or {}).get("1h", 0.0),
                "observed_at": datetime.fromtimestamp(current["dt"], tz=timezone.utc).isoformat(),
            },
            "outlook": {
                "days": len(daily),
                "temp_min": round(min(temps_min), 1),
                "temp_max": round(max(temps_max), 1),
                "temp_avg": round((sum(temps_max) + sum(temps_min)) / (2 * len(temps_max)), 1),
                "avg_humidity": avg_humidity,
                "total_rain_mm": total_rain,
                "rainy_days": sum(1 for d in daily if d["rain_mm"] >= 2.5),
            },
            "daily": daily,
            "alerts": alerts,
            "good_spray_days": spray_days[:3],
            "source": "OpenWeatherMap",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }


weather_service = WeatherService()
