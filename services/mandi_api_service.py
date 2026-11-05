"""Real-time mandi (market) prices.

Sources, tried in this order:
1. Mandi Prices API  - https://mandi-api.onrender.com/v1/prices?state=..&commodity=..
   (latest Agmarknet prices; filters: state, commodity, market; max 200 rows;
   the district filter is ignored, so we filter districts ourselves)
2. data.gov.in       - "Current Daily Price of Various Commodities from Various Markets"
   (needs MANDI_API_KEY; the public sample key returns only 10 rows)
3. The project's mandi CSV dataset (handled by the caller) when both are down.
"""

import asyncio
import re
import time

import httpx

from core.config import settings

# Farmer crop name (lower case) -> commodity name used by the Agmarknet APIs
COMMODITY_NAMES = {
    "onion": "Onion", "kanda": "Onion", "grapes": "Grapes", "grape": "Grapes",
    "pomegranate": "Pomegranate", "anar": "Pomegranate", "wheat": "Wheat",
    "chickpea": "Bengal Gram(Gram)(Whole)", "chana": "Bengal Gram(Gram)(Whole)", "gram": "Bengal Gram(Gram)(Whole)",
    "tur": "Red gram/Arhar/Tur(whole)", "arhar": "Red gram/Arhar/Tur(whole)", "pigeon pea": "Red gram/Arhar/Tur(whole)",
    "soybean": "Soyabean", "soyabean": "Soyabean", "cotton": "Cotton", "maize": "Maize",
    "rice": "Paddy(Common)", "paddy": "Paddy(Common)", "tomato": "Tomato", "potato": "Potato",
    "groundnut": "Groundnut", "mustard": "Mustard", "jowar": "Jowar(Sorghum)", "sorghum": "Jowar(Sorghum)",
    "bajra": "Bajra(Pearl Millet/Cumbu)", "green chilli": "Green Chilli", "chilli": "Green Chilli",
    "banana": "Banana", "moong": "Green Gram(Moong)(Whole)", "green gram": "Green Gram(Moong)(Whole)",
    "urad": "Black Gram(Urd Beans)(Whole)", "black gram": "Black Gram(Urd Beans)(Whole)",
    "sugarcane": "Sugarcane",
}

# Old district name -> name used in the latest Agmarknet data
DISTRICT_ALIASES = {
    "ahmednagar": "ahilyanagar", "aurangabad": "chattrapati sambhajinagar", "osmanabad": "dharashiv",
    "amravati": "amarawati", "gondia": "gondiya", "sholapur": "solapur",
}


def _key(value):
    return re.sub(r"[^a-z]+", " ", str(value or "").lower()).strip()


def commodity_name(crop):
    """'Chickpea (Chana)' -> 'Bengal Gram(Gram)(Whole)'. Unknown names are returned as typed."""
    text = _key(crop)
    if text in COMMODITY_NAMES:
        return COMMODITY_NAMES[text]
    for alias in sorted(COMMODITY_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return COMMODITY_NAMES[alias]
    return str(crop or "").split(" (")[0].strip()


def same_district(a, b):
    a, b = _key(a), _key(b)
    if not a or not b:
        return False
    a, b = DISTRICT_ALIASES.get(a, a), DISTRICT_ALIASES.get(b, b)
    return a == b or a.startswith(b) or b.startswith(a)


MAX_PARALLEL = 3  # the free Mandi Prices API answers 429 when we send too many requests at once


class MandiApiService:
    def __init__(self):
        self.live_url = settings.MANDI_LIVE_API_URL
        self.api_key = settings.MANDI_API_KEY
        self.resource_id = settings.MANDI_RESOURCE_ID
        self.base_url = "https://api.data.gov.in/resource/"
        self._cache = {}
        self._down_until = {}
        self.last_error = None
        self._semaphores = {}

    def _semaphore(self):
        loop = asyncio.get_running_loop()
        if id(loop) not in self._semaphores:
            self._semaphores[id(loop)] = asyncio.Semaphore(MAX_PARALLEL)
        return self._semaphores[id(loop)]

    async def _api_get(self, url, params):
        """GET with a concurrency limit and polite retries on 429 (rate limit)."""
        async with self._semaphore():
            async with httpx.AsyncClient(timeout=settings.MANDI_LIVE_TIMEOUT_SECONDS) as client:
                for attempt in range(4):
                    response = await client.get(url, params=params)
                    if response.status_code != 429:
                        response.raise_for_status()
                        return response.json()
                    wait = float(response.headers.get("Retry-After") or 0) or 1.5 * (attempt + 1)
                    await asyncio.sleep(min(wait, 8))
                response.raise_for_status()

    @property
    def enabled(self):
        return bool(self.live_url or self.api_key)

    @property
    def using_sample_key(self):
        return self.api_key == settings.MANDI_SAMPLE_KEY

    # ------------------------------------------------------------ helpers
    async def _cached(self, source, cache_key, fetch):
        """Cache results for MANDI_API_CACHE_SECONDS; after a failure skip the source for 2 minutes."""
        hit = self._cache.get((source, cache_key))
        if hit and time.monotonic() - hit[0] <= settings.MANDI_API_CACHE_SECONDS:
            return [dict(r) for r in hit[1]]
        if time.monotonic() < self._down_until.get(source, 0):
            raise RuntimeError(f"{source} recently failed ({self.last_error}); retrying later")
        try:
            rows = await fetch()
        except Exception as exc:
            self.last_error = f"{source}: {type(exc).__name__} {str(exc).splitlines()[0] if str(exc) else ''}".strip()
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is None or status >= 500:  # server down / timeout -> pause this source
                self._down_until[source] = time.monotonic() + 120
            raise
        self._cache[(source, cache_key)] = (time.monotonic(), [dict(r) for r in rows])
        return rows

    # ------------------------------------------------------- source 1
    async def fetch_live(self, state, commodity="", market=""):
        """Latest prices from the Mandi Prices API (max 200 rows)."""
        if not self.live_url:
            return []
        params = {k: v for k, v in {"state": state, "commodity": commodity, "market": market}.items() if v}

        async def fetch():
            body = await self._api_get(self.live_url, params)
            rows = body.get("data") or []
            for row in rows:
                row["market"] = str(row.get("market") or "").strip()
                row["source"] = "mandi-api"
            return rows

        return await self._cached("mandi-api", tuple(sorted(params.items())), fetch)

    async def _get_json(self, path, params, cache_seconds):
        """GET {api root}/{path} with caching; returns the 'data' field."""
        root = self.live_url.rsplit("/prices", 1)[0]
        key = ("mandi-api:" + path, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] <= cache_seconds:
            return hit[1]
        data = (await self._api_get(f"{root}/{path}", params)).get("data") or []
        self._cache[key] = (time.monotonic(), data)
        return data

    async def get_states(self):
        return await self._get_json("states", {}, 6 * 3600)

    async def get_markets(self, state):
        """[{market, district}, ...] for a state (the API's list of reporting mandis)."""
        rows = await self._get_json("markets", {"state": state}, 6 * 3600)
        return [{"market": str(r.get("market") or "").strip(), "district": str(r.get("district") or "").strip()}
                for r in rows if r.get("market")]

    async def get_history(self, state, commodity, market=""):
        """Daily history. With a market: raw rows for that mandi (MARKET_SPECIFIC);
        without: one state-average row per day (avg_modal_price ...)."""
        params = {k: v for k, v in {"state": state, "commodity": commodity, "market": market}.items() if v}
        return await self._get_json("prices/history", params, settings.MANDI_API_CACHE_SECONDS)

    # ------------------------------------------------------- source 2
    async def fetch_data_gov(self, state, district="", market="", commodity="", limit=500, timeout=None):
        if not self.api_key:
            return []
        filters = {k: v for k, v in {"state.keyword": state, "district": district, "market": market,
                                      "commodity": commodity}.items() if v}

        async def fetch():
            async with httpx.AsyncClient(timeout=timeout or settings.MANDI_LIVE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{self.base_url}{self.resource_id}", params={
                    "api-key": self.api_key, "format": "json", "limit": limit,
                    **{f"filters[{k}]": v for k, v in filters.items()},
                })
                response.raise_for_status()
                rows = response.json().get("records") or []
                for row in rows:
                    row["source"] = "data.gov.in"
                return rows

        return await self._cached("data.gov.in", (tuple(sorted(filters.items())), limit), fetch)

    # ------------------------------------------------------- public API
    async def get_market_prices(self, state: str, district: str, market: str, commodity: str):
        """Live records for the market prediction form.

        Returns rows for the district (or market) when available, else the whole state.
        Returns [] when no live source answers - the caller then uses the dataset.
        """
        name = commodity_name(commodity)
        try:
            rows = await self.fetch_live(state, name)
        except Exception as exc:
            print(f"Mandi API unavailable: {exc}")
            rows = []
        if not rows:
            try:
                rows = await self.fetch_data_gov(state, district, "", name, limit=100,
                                                 timeout=settings.MANDI_API_TIMEOUT_SECONDS)
            except Exception as exc:
                print(f"data.gov.in unavailable: {exc}")
                rows = []
        if not rows:
            return []
        if market:
            in_market = [r for r in rows if _key(market).split(" apmc")[0] in _key(r.get("market"))]
            if in_market:
                return in_market
        in_district = [r for r in rows if same_district(r.get("district"), district)]
        return in_district or rows

    async def get_farmer_prices(self, state, district, crops):
        """Prices for one farmer: everything traded in their district today, plus the
        best markets in their state for each of their own crops.

        Returns (rows, source) where rows are raw API records with extra flags.
        """
        names = list(dict.fromkeys(commodity_name(c) for c in crops if c))
        calls = [self.fetch_live(state)] + [self.fetch_live(state, n) for n in names]
        results = await asyncio.gather(*calls, return_exceptions=True)
        rows, source = [], "Live - Mandi Prices API (Agmarknet)"
        for result in results:
            if isinstance(result, list):
                rows += result
            else:
                print(f"Mandi API call failed: {result}")

        if not rows:
            try:
                rows = await self.fetch_data_gov(state, district)
                source = "Live - data.gov.in (Agmarknet)"
            except Exception as exc:
                print(f"data.gov.in unavailable: {exc}")
                return [], None

        picked, seen = [], set()
        for row in rows:
            key = (row.get("market"), row.get("commodity"), row.get("variety"), row.get("arrival_date"))
            if key in seen:
                continue
            seen.add(key)
            row["in_my_district"] = same_district(row.get("district"), district)
            row["is_my_crop"] = row.get("commodity") in names
            if row["in_my_district"]:
                picked.append(row)

        # For the farmer's crops that are not traded in their district, show the best nearby markets.
        for name in names:
            if any(r["commodity"] == name for r in picked):
                continue
            others = sorted((r for r in rows if r.get("commodity") == name and not r.get("in_my_district")),
                            key=lambda r: -float(r.get("modal_price") or 0))
            picked += others[:3]
        return picked, source


mandi_api_service = MandiApiService()
