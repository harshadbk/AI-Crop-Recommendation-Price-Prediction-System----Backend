"""Monthly mandi price forecasting built on Agriculture_price_dataset.csv.

Approach (kept deliberately simple and explainable):
1. Aggregate ~740k daily mandi rows into a monthly median modal price per
   (state, district, market, commodity). The aggregate is cached on disk.
2. Pick the most specific series with enough history:
   market -> district -> state -> all-India for the commodity.
3. Fit a damped-trend Holt model on deseasonalised log prices (parameters
   chosen by one-step-ahead error) and blend it 50/50 with a seasonal-naive
   forecast. Seasonal month factors come from the all-India commodity
   series, shrunk halfway toward 1.
4. Backtest on the last few months to report MAPE / accuracy, and give an
   80% prediction band from the in-sample residuals.
"""

import math
from threading import RLock

import numpy as np
import pandas as pd

from core.config import settings

MIN_MONTHS = 8
HORIZON = 3
BACKTEST_MONTHS = 3
Z_80 = 1.2816


def _key(value):
    import re

    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _holt_damped(log_values, alpha, beta, phi):
    """Return one-step-ahead fitted values plus the final level and trend."""
    level = log_values[0]
    trend = log_values[1] - log_values[0] if len(log_values) > 1 else 0.0
    fitted = [level]
    for value in log_values[1:]:
        forecast = level + phi * trend
        fitted.append(forecast)
        new_level = alpha * value + (1 - alpha) * forecast
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    return np.array(fitted), level, trend


def _project(level, trend, phi, horizon):
    out, damp = [], 0.0
    for h in range(1, horizon + 1):
        damp += phi ** h
        out.append(level + damp * trend)
    return np.array(out)


def _fit(log_values):
    best = None
    for alpha in (0.2, 0.35, 0.5, 0.7, 0.9):
        for beta in (0.05, 0.15, 0.3):
            for phi in (0.8, 0.9, 0.98):
                fitted, level, trend = _holt_damped(log_values, alpha, beta, phi)
                errors = log_values[2:] - fitted[2:]
                sse = float(np.sum(errors ** 2)) if len(errors) else 0.0
                if best is None or sse < best[0]:
                    best = (sse, alpha, beta, phi, level, trend, errors)
    _, alpha, beta, phi, level, trend, errors = best
    sigma = float(np.std(errors, ddof=1)) if len(errors) > 2 else 0.08
    return {"alpha": alpha, "beta": beta, "phi": phi, "level": level, "trend": trend, "sigma": max(sigma, 0.02)}


class PriceForecastService:
    def __init__(self):
        self._lock = RLock()
        self._monthly = None
        self._seasonal = {}
        self._cache = {}

    # ---------- data ----------
    def _build_monthly(self):
        path = settings.DATA_DIR / "Agriculture_price_dataset.csv"
        df = pd.read_csv(
            path,
            usecols=["STATE", "District Name", "Market Name", "Commodity", "Modal_Price", "Price Date"],
        )
        df["month"] = pd.to_datetime(df["Price Date"], format="%m/%d/%Y", errors="coerce").dt.to_period("M")
        df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
        df = df.dropna(subset=["month", "Modal_Price"])
        df = df[df["Modal_Price"] > 0]
        for src, dst in [("STATE", "state"), ("District Name", "district"), ("Market Name", "market"), ("Commodity", "commodity")]:
            df[dst] = df[src].astype(str).map(_key)
        return (
            df.groupby(["state", "district", "market", "commodity", "month"])["Modal_Price"]
            .agg(price="median", n="size")
            .reset_index()
        )

    def _load(self):
        with self._lock:
            if self._monthly is None:
                from services.agri_data_service import get_agri_data_service

                source = settings.DATA_DIR / "Agriculture_price_dataset.csv"
                if not source.exists():
                    self._monthly = pd.DataFrame()
                else:
                    self._monthly = get_agri_data_service()._read_pickle_cache(
                        source, "price_monthly_v1.pkl", self._build_monthly
                    )
            return self._monthly

    def warm_up(self):
        self._load()

    # ---------- series selection ----------
    @staticmethod
    def _to_series(frame):
        # Weight each market's monthly median by its number of observations.
        grouped = frame.assign(w=frame["price"] * frame["n"]).groupby("month")[["w", "n"]].sum()
        series = (grouped["w"] / grouped["n"]).sort_index()
        full = pd.period_range(series.index.min(), series.index.max(), freq="M")
        return series.reindex(full).interpolate(limit_direction="both")

    def _match(self, column, frame, value):
        key = _key(value)
        if not key:
            return frame.iloc[0:0]
        exact = frame[frame[column] == key]
        if not exact.empty:
            return exact
        return frame[frame[column].str.contains(key, regex=False, na=False)]

    def _select_series(self, state, district, market, commodity):
        df = self._load()
        if df.empty or not commodity:
            return None, None
        crop = self._match("commodity", df, commodity)
        if crop.empty:
            return None, None

        levels = []
        st = self._match("state", crop, state) if state else crop.iloc[0:0]
        ds = self._match("district", st, district) if district else st.iloc[0:0]
        scope = ds if not ds.empty else st if not st.empty else crop
        mk = self._match("market", scope, market) if market else crop.iloc[0:0]
        levels += [("market", mk), ("district", ds), ("state", st), ("india", crop)]

        for level, frame in levels:
            if frame.empty:
                continue
            series = self._to_series(frame)
            if series.notna().sum() >= MIN_MONTHS:
                return level, series
        return None, None

    def _seasonal_factors(self, commodity, end=None):
        """Month-of-year factors from the all-India series, using data up to `end` only."""
        key = (_key(commodity), str(end))
        if key in self._seasonal:
            return self._seasonal[key]
        df = self._load()
        crop = self._match("commodity", df, commodity)
        factors = {m: 1.0 for m in range(1, 13)}
        if not crop.empty:
            series = self._to_series(crop)
            if end is not None:
                series = series[series.index <= end]
            if len(series) >= 18:
                baseline = series.rolling(12, center=True, min_periods=6).mean()
                ratio = (series / baseline).dropna()
                by_month = ratio.groupby(ratio.index.month).median()
                # Shrink halfway toward 1: only ~2 years of history (best in backtests).
                for month, value in by_month.items():
                    factors[int(month)] = float(1 + 0.5 * (value - 1))
        self._seasonal[key] = factors
        return factors

    # ---------- forecasting ----------
    def _forecast_values(self, series, seasonal, horizon):
        months = series.index
        deseason = np.array([series.iloc[i] / seasonal[months[i].month] for i in range(len(series))])
        log_values = np.log(deseason)
        model = _fit(log_values)
        base = _project(model["level"], model["trend"], model["phi"], horizon)
        future = [months[-1] + h for h in range(1, horizon + 1)]
        holt = [math.exp(b) * seasonal[p.month] for b, p in zip(base, future)]
        last_month = months[-1].month
        seasonal_naive = [series.iloc[-1] * seasonal[p.month] / seasonal[last_month] for p in future]
        # Equal blend of trend model and seasonal naive: lowest backtest error on this dataset.
        values = [float((h + n) / 2) for h, n in zip(holt, seasonal_naive)]
        return future, values, model

    def _backtest(self, series, commodity):
        if len(series) < MIN_MONTHS + BACKTEST_MONTHS:
            return None
        train, test = series.iloc[:-BACKTEST_MONTHS], series.iloc[-BACKTEST_MONTHS:]
        seasonal = self._seasonal_factors(commodity, end=train.index[-1])  # no look-ahead
        _, predicted, _ = self._forecast_values(train, seasonal, BACKTEST_MONTHS)
        errors = [abs(p - a) / a for p, a in zip(predicted, test.values) if a]
        return float(np.mean(errors)) if errors else None

    def forecast(self, state, district, market, commodity, horizon=HORIZON):
        cache_key = (_key(state), _key(district), _key(market), _key(commodity), horizon)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            level, series = self._select_series(state, district, market, commodity)
        except Exception as exc:
            print(f"Price forecast unavailable: {exc}")
            level, series = None, None
        if series is None:
            self._cache[cache_key] = None
            return None

        seasonal = self._seasonal_factors(commodity)
        future, values, model = self._forecast_values(series, seasonal, horizon)
        mape = self._backtest(series, commodity)
        last_price = float(series.iloc[-1])
        change_pct = (values[0] - last_price) / last_price * 100 if last_price else 0.0

        points = []
        for h, (period, value) in enumerate(zip(future, values), start=1):
            spread = Z_80 * model["sigma"] * math.sqrt(h)
            points.append({
                "month": period.strftime("%b %Y"),
                "price": int(round(value)),
                "low": int(round(value * math.exp(-spread))),
                "high": int(round(value * math.exp(spread))),
            })

        history = [
            {"month": p.strftime("%b %Y"), "price": int(round(v))}
            for p, v in zip(series.index[-12:], series.values[-12:])
        ]

        result = {
            "level": level,
            "months_of_history": int(len(series)),
            "last_actual_month": series.index[-1].strftime("%b %Y"),
            "last_actual_price": int(round(last_price)),
            "next_month_price": points[0]["price"],
            "change_pct": round(change_pct, 1),
            "points": points,
            "history": history,
            "backtest_mape_pct": round(mape * 100, 1) if mape is not None else None,
            "accuracy_pct": round(max(0.0, 100 - mape * 100), 1) if mape is not None else None,
            "confidence": (
                "High" if mape is not None and mape < 0.08 and level in ("market", "district")
                else "Medium" if mape is not None and mape < 0.15
                else "Low"
            ),
            "method": "Blend of damped Holt trend and seasonal-naive on monthly median mandi prices",
        }
        self._cache[cache_key] = result
        return result


price_forecast_service = PriceForecastService()
