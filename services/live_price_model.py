"""Real-time mandi price model (daily), built on the live Mandi Prices API.

Steps
1. Snapshot: latest prices in every reporting mandi of the state for the commodity
   -> state median, best mandis to sell, the farmer's own district mandis, price spread.
2. History: daily prices for the chosen mandi (or the state average) from the API,
   merged with our own stored history in MongoDB (it grows every day).
3. Forecast: an ENSEMBLE of four models on log prices -
     * Holt damped trend        (follows the trend, slowly flattens it)
     * Recent linear trend      (last 14 trading days)
     * Ridge regression         (machine learning on lag features: last 1/2/3 days,
                                 7-day mean, 7-day volatility, 7-day momentum)
     * Naive                    (tomorrow = today; the baseline every model must beat)
   Each model is tested on the last few days it has NOT seen (rolling-origin backtest);
   weights = inverse of its error, so the most accurate model counts most.
4. Advice: sell now / hold / sell at a better mandi, from the forecast and the snapshot.
"""

import asyncio
import math
from datetime import date, datetime, timedelta

import numpy as np
from fastapi.concurrency import run_in_threadpool

from services import price_store
from services.mandi_api_service import commodity_name, mandi_api_service, same_district
from services.price_forecast_service import _fit, _project

Z_80 = 1.2816
MIN_POINTS = 8


# ----------------------------------------------------------------- helpers
def _date(value):
    text = str(value or "")[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _next_trading_days(start, n):
    days, d = [], start
    while len(days) < n:
        d += timedelta(days=1)
        if d.weekday() != 6:  # mandis are mostly closed on Sunday
            days.append(d)
    return days


def daily_series(rows, price_key="modal_price"):
    """Rows -> sorted [(date, median price)] with one value per day."""
    by_day = {}
    for r in rows:
        d, p = _date(r.get("arrival_date")), r.get(price_key)
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if d and p > 0:
            by_day.setdefault(d, []).append(p)
    return sorted((d, float(np.median(v))) for d, v in by_day.items())


# ----------------------------------------------------------------- models
def _m_naive(y, h):
    return np.full(h, y[-1])


def _m_holt(y, h):
    fit = _fit(y)
    return _project(fit["level"], fit["trend"], fit["phi"], h)


def _m_trend(y, h):
    w = y[-14:]
    if len(w) < 4:
        return _m_naive(y, h)
    slope, intercept = np.polyfit(np.arange(len(w)), w, 1)
    damp = np.cumsum(0.9 ** np.arange(1, h + 1))
    return y[-1] + slope * damp


def _lag_features(y, t):
    """Features to predict the change from day t-1 to day t, using data up to t-1."""
    past = y[:t]
    last7 = past[-7:]
    return [past[-1] - past[-2], past[-2] - past[-3], past[-1] - past[-4] if len(past) >= 4 else 0.0,
            past[-1] - float(np.mean(last7)), float(np.std(last7)), past[-1] - past[-min(7, len(past))]]


def _m_ridge(y, h):
    if len(y) < 15:
        return _m_naive(y, h)
    from sklearn.linear_model import Ridge

    X = [_lag_features(y, t) for t in range(4, len(y))]
    target = [y[t] - y[t - 1] for t in range(4, len(y))]
    model = Ridge(alpha=1.0).fit(np.array(X), np.array(target))
    path = list(y)
    for _ in range(h):  # recursive multi-step forecast
        change = float(model.predict(np.array([_lag_features(np.array(path), len(path))]))[0])
        path.append(path[-1] + float(np.clip(change, -0.15, 0.15)))
    return np.array(path[len(y):])


MODELS = {
    "Holt damped trend": _m_holt,
    "Recent linear trend": _m_trend,
    "Ridge regression (lag features)": _m_ridge,
    "Naive (no change)": _m_naive,
}


def ensemble_forecast(series, horizon=7):
    """series: [(date, price)] -> forecast dict, or None if there is too little data."""
    if len(series) < MIN_POINTS:
        return None
    dates = [d for d, _ in series]
    y = np.log(np.array([p for _, p in series]))
    n = len(y)

    # rolling-origin backtest: predict each of the last k days from the data before it
    k = min(7, max(3, n // 5))
    errors = {name: [] for name in MODELS}
    moves = []
    for origin in range(n - k, n):
        train = y[:origin]
        preds = {}
        for name, fn in MODELS.items():
            try:
                preds[name] = float(fn(train, 1)[0])
            except Exception:
                preds[name] = float(train[-1])
            errors[name].append(abs(math.exp(preds[name] - y[origin]) - 1))
        moves.append((train[-1], y[origin], preds))

    mape = {name: float(np.mean(e)) * 100 for name, e in errors.items()}
    raw = {name: 1.0 / (m + 0.5) ** 2 for name, m in mape.items()}
    total = sum(raw.values())
    weights = {name: w / total for name, w in raw.items()}

    ens_err, hits, moved = [], 0, 0
    for prev, actual, preds in moves:
        pred = sum(weights[nm] * preds[nm] for nm in MODELS)
        ens_err.append(abs(math.exp(pred - actual) - 1))
        if abs(actual - prev) > 1e-6:
            moved += 1
            hits += int(np.sign(pred - prev) == np.sign(actual - prev))
    resid = np.array([math.log1p(e) for e in ens_err])
    sigma = max(float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.05, 0.02)

    paths = {name: fn(y, horizon) for name, fn in MODELS.items()}
    combined = sum(weights[nm] * np.asarray(paths[nm]) for nm in MODELS)
    future = _next_trading_days(dates[-1], horizon)
    points = []
    for i, (d, v) in enumerate(zip(future, combined), start=1):
        spread = Z_80 * sigma * math.sqrt(i)
        points.append({"date": d.isoformat(), "label": d.strftime("%d %b"), "price": int(round(math.exp(v))),
                       "low": int(round(math.exp(v - spread))), "high": int(round(math.exp(v + spread)))})

    last = float(math.exp(y[-1]))
    ens_mape = float(np.mean(ens_err)) * 100
    return {
        "last_date": dates[-1].isoformat(),
        "last_price": int(round(last)),
        "next_price": points[0]["price"],
        "end_price": points[-1]["price"],
        "change_pct": round((points[-1]["price"] / last - 1) * 100, 1),
        "points": points,
        "horizon_days": horizon,
        "models": sorted(({"name": nm, "backtest_error_pct": round(mape[nm], 2), "weight_pct": round(weights[nm] * 100, 1)}
                          for nm in MODELS), key=lambda m: -m["weight_pct"]),
        "backtest_error_pct": round(ens_mape, 2),
        "naive_error_pct": round(mape["Naive (no change)"], 2),
        "accuracy_pct": round(max(0.0, 100 - ens_mape), 1),
        "direction_accuracy_pct": round(hits / moved * 100, 1) if moved else None,
        "backtest_days": k,
        "training_days": n,
        "method": "Weighted ensemble: Holt damped trend + linear trend + Ridge regression on lag features + naive",
    }


# --------------------------------------------------------------- snapshot
def market_snapshot(rows, district=""):
    """Latest price per mandi (median over varieties on its latest day)."""
    by_market = {}
    for r in rows:
        d = _date(r.get("arrival_date"))
        name = str(r.get("market") or "").strip()
        if not d or not name:
            continue
        slot = by_market.setdefault(name, {"market": name, "district": r.get("district"), "date": d, "prices": [],
                                           "min": [], "max": []})
        if d > slot["date"]:
            slot.update(date=d, prices=[], min=[], max=[])
        if d == slot["date"]:
            slot["prices"].append(float(r.get("modal_price") or 0))
            slot["min"].append(float(r.get("min_price") or 0))
            slot["max"].append(float(r.get("max_price") or 0))
    if not by_market:
        return None
    newest = max(m["date"] for m in by_market.values())
    markets = []
    for m in by_market.values():
        if (newest - m["date"]).days > 7 or not m["prices"]:
            continue  # ignore mandis that have not reported this week
        markets.append({"market": m["market"], "district": m["district"], "date": m["date"].isoformat(),
                        "modal_price": int(np.median(m["prices"])), "min_price": int(min(m["min"])),
                        "max_price": int(max(m["max"])), "in_my_district": same_district(m["district"], district)})
    if not markets:
        return None
    markets.sort(key=lambda m: -m["modal_price"])
    prices = [m["modal_price"] for m in markets]
    median = float(np.median(prices))
    mine = [m for m in markets if m["in_my_district"]]
    best_local = max(mine, key=lambda m: m["modal_price"]) if mine else None
    return {
        "as_of": newest.isoformat(),
        "reporting_markets": len(markets),
        "state_median": int(median),
        "spread_pct": round((max(prices) - min(prices)) / median * 100, 1) if median else 0,
        "best_markets": markets[:5],
        "district_markets": mine[:8],
        "best_in_district": best_local,
        "gain_vs_district": (markets[0]["modal_price"] - best_local["modal_price"]) if best_local else None,
    }


def _advice(forecast, snapshot, commodity):
    tips = []
    if forecast:
        c = forecast["change_pct"]
        if c >= 3:
            tips.append(f"{commodity} is expected to rise about {c}% in the next {forecast['horizon_days']} market days - "
                        "hold stock if you have safe storage.")
        elif c <= -3:
            tips.append(f"{commodity} is expected to fall about {abs(c)}% in the next {forecast['horizon_days']} market days - "
                        "selling soon is safer.")
        else:
            tips.append(f"{commodity} prices look stable for the next {forecast['horizon_days']} market days.")
        if forecast["backtest_error_pct"] > 15:
            tips.append("Prices have been very volatile, so treat the forecast as a rough guide.")
    if snapshot:
        best = snapshot["best_markets"][0]
        local = snapshot.get("best_in_district")
        if local and snapshot.get("gain_vs_district", 0) > 0:
            tips.append(f"Best price today: {best['market']} ({best['district']}) at Rs {best['modal_price']:,}/q - "
                        f"Rs {snapshot['gain_vs_district']:,} more than {local['market']} in your district. "
                        "Check transport cost before moving.")
        elif local:
            tips.append(f"{local['market']} in your district is already among the best prices (Rs {local['modal_price']:,}/q).")
        else:
            tips.append(f"Best price today: {best['market']} ({best['district']}) at Rs {best['modal_price']:,}/q.")
    return tips


# ------------------------------------------------------------------ public
async def analyze(state, commodity, district="", market="", horizon=7, with_history=True):
    """Full live analysis for one commodity. Returns None if the live API has nothing.
    with_history=False -> only today's snapshot (one API call), used to price many candidate crops."""
    name = commodity_name(commodity)
    try:
        latest = await mandi_api_service.fetch_live(state, name)
    except Exception as exc:
        print(f"Live price model: snapshot unavailable ({exc})")
        latest = []
    snapshot = market_snapshot(latest, district)
    if not with_history:
        await run_in_threadpool(price_store.save_rows, latest)
        if not snapshot:
            return None
        return {"commodity": name, "state": state, "district": district, "level": "snapshot", "market": None,
                "series": [], "forecast": None, "snapshot": snapshot, "advice": _advice(None, snapshot, name),
                "source": "Mandi Prices API (Agmarknet)", "generated_at": datetime.now().isoformat(timespec="seconds")}

    # history for the most useful mandi: the one asked for, else the busiest one in the farmer's district
    target = ""
    if market:
        wanted = market.lower().replace(" apmc", "").strip()
        target = next((r["market"].strip() for r in latest if wanted and wanted in r["market"].lower()), market.strip())
    elif snapshot and snapshot["district_markets"]:
        counts = {}
        for r in latest:
            if same_district(r.get("district"), district):
                counts[r["market"].strip()] = counts.get(r["market"].strip(), 0) + 1
        target = max(counts, key=counts.get) if counts else ""

    history_rows, level = [], "state"
    if target:
        try:
            history_rows = await mandi_api_service.get_history(state, name, target)
            level = "market"
        except Exception as exc:
            print(f"Live price model: market history unavailable ({exc})")

    # save everything we saw, then read back our own (growing) stored history
    await run_in_threadpool(price_store.save_rows, list(history_rows) + list(latest))
    stored = await run_in_threadpool(price_store.load_rows, state, name, target or None) if target else []
    merged = {r.get("id"): r for r in list(stored) + list(history_rows) if r.get("id") is not None}
    series = daily_series(merged.values())

    if len(series) < MIN_POINTS:
        try:
            avg = await mandi_api_service.get_history(state, name)
            series, level, target = daily_series(avg, "avg_modal_price"), "state", ""
        except Exception as exc:
            print(f"Live price model: state history unavailable ({exc})")

    if not series and not snapshot:
        return None
    forecast = await run_in_threadpool(ensemble_forecast, series, horizon) if series else None
    return {
        "commodity": name,
        "state": state,
        "district": district,
        "level": level,
        "market": target or None,
        "series": [{"date": d.isoformat(), "label": d.strftime("%d %b"), "price": int(round(p))} for d, p in series[-60:]],
        "forecast": forecast,
        "snapshot": snapshot,
        "advice": _advice(forecast, snapshot, name),
        "source": "Mandi Prices API (Agmarknet) + stored history",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


async def analyze_many(state, commodities, district=""):
    results = await asyncio.gather(*(analyze(state, c, district) for c in commodities), return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def chart_series(result, history_days=21):
    """History + forecast in the {label, price} format used by TrendChart."""
    out = [{"label": p["label"], "price": p["price"]} for p in (result.get("series") or [])[-history_days:]]
    for p in (result.get("forecast") or {}).get("points", []):
        out.append({"label": f"{p['label']} (f)", "price": p["price"], "low": p["low"], "high": p["high"], "is_forecast": True})
    return out


def latest_price(result, district=""):
    """Best single 'current price' for revenue estimates: district median, else state median."""
    snap = (result or {}).get("snapshot")
    if not snap:
        return None
    local = [m["modal_price"] for m in snap["district_markets"]]
    if local:
        return {"price": int(np.median(local)), "scope": "district", "as_of": snap["as_of"]}
    return {"price": snap["state_median"], "scope": "state", "as_of": snap["as_of"]}


def today():
    return date.today().isoformat()
