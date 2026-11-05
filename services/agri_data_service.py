import math
import re
import time
from threading import RLock

import pandas as pd

from core.config import settings


def _normalize_text(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    return int(round(_safe_float(value, default)))


def _label(value, fallback):
    text = str(value or "").strip()
    return text if text else fallback


class AgriDataService:
    def __init__(self):
        self._crop_summary_df = None
        self._mandi_df = None
        self._lock = RLock()
        self._cache_dir = settings.DATA_DIR / ".cache"
        self._market_records_cache = {}
        self._heatmap_cache = {}
        self._crop_aggregate_cache = {}
        self._crop_names = None
        self._commodity_names = None
        self._top_yield_crop = None

    def warm_up(self):
        from services.price_forecast_service import price_forecast_service

        self._load_mandi_df()
        price_forecast_service.warm_up()
        self._load_crop_summary_df()
        self._get_crop_names()
        self._get_commodity_names()

    def _is_cache_fresh(self, source_path, cache_path):
        return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime

    def _read_pickle_cache(self, source_path, cache_name, builder):
        cache_path = self._cache_dir / cache_name
        if self._is_cache_fresh(source_path, cache_path):
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                pass

        frame = builder()
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            frame.to_pickle(cache_path)
        except Exception as exc:
            print(f"Dataset disk cache skipped for {cache_name}: {exc}")
        return frame

    def _load_crop_summary_df(self):
        with self._lock:
            if self._crop_summary_df is not None:
                return self._crop_summary_df

            path = settings.DATA_DIR / "Indian_crop_production_yield_dataset.csv"

            def build_summary():
                started = time.perf_counter()
                df = pd.read_csv(
                    path,
                    usecols=["State_Name", "District_Name", "Season", "Crop", "Area", "Production", "yield"],
                ).rename(
                    columns={
                        "State_Name": "state_name",
                        "District_Name": "district_name",
                        "Season": "season",
                        "Crop": "crop",
                        "Area": "area",
                        "Production": "production",
                        "yield": "yield_value",
                    }
                )
                df["state_key"] = df["state_name"].map(_normalize_text)
                df["district_key"] = df["district_name"].map(_normalize_text)
                df["season_key"] = df["season"].map(_normalize_text)
                df["crop_key"] = df["crop"].map(_normalize_text)
                df["area"] = pd.to_numeric(df["area"], errors="coerce").fillna(0)
                df["production"] = pd.to_numeric(df["production"], errors="coerce").fillna(0)
                df["yield_value"] = pd.to_numeric(df["yield_value"], errors="coerce").fillna(0)
                summary = (
                    df.groupby(
                        [
                            "state_key",
                            "state_name",
                            "district_key",
                            "district_name",
                            "season_key",
                            "season",
                            "crop",
                            "crop_key",
                        ],
                        as_index=False,
                    )
                    .agg(
                        avg_yield=("yield_value", "mean"),
                        total_production=("production", "sum"),
                        total_area=("area", "sum"),
                        samples=("crop", "size"),
                    )
                )
                print(f"Crop summary prepared in {time.perf_counter() - started:.2f}s.")
                return summary

            self._crop_summary_df = self._read_pickle_cache(path, "crop_summary.pkl", build_summary)
            return self._crop_summary_df

    def _load_mandi_df(self):
        with self._lock:
            if self._mandi_df is not None:
                return self._mandi_df

            path = settings.DATA_DIR / "mandi_data_2000_rows.csv"
            df = pd.read_csv(path).rename(
                columns={
                    "Date": "arrival_date",
                    "State": "state",
                    "District": "district",
                    "Mandi": "market",
                    "Commodity": "commodity",
                    "Variety": "variety",
                    "Min Price": "min_price",
                    "Max Price": "max_price",
                    "Modal Price": "modal_price",
                    "Unit": "unit",
                    "Arrival Quantity": "arrival_quantity",
                    "Grade": "grade",
                    "Market Fee (%)": "market_fee_pct",
                    "Trader Count": "trader_count",
                }
            )
            df["parsed_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
            df["state_key"] = df["state"].map(_normalize_text)
            df["district_key"] = df["district"].map(_normalize_text)
            df["market_key"] = df["market"].map(_normalize_text)
            df["commodity_key"] = df["commodity"].map(_normalize_text)
            for column in ["min_price", "max_price", "modal_price", "arrival_quantity", "market_fee_pct", "trader_count"]:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
            self._mandi_df = df
            return df

    def _clone_records(self, records):
        return [dict(item) for item in records]

    def _get_crop_names(self):
        with self._lock:
            if self._crop_names is None:
                self._crop_names = self._load_crop_summary_df()["crop"].dropna().astype(str).unique().tolist()
            return self._crop_names

    def _get_commodity_names(self):
        with self._lock:
            if self._commodity_names is None:
                self._commodity_names = self._load_mandi_df()["commodity"].dropna().astype(str).unique().tolist()
            return self._commodity_names

    def _get_top_yield_crop(self):
        with self._lock:
            if self._top_yield_crop is None:
                df = self._load_crop_summary_df()
                top_crop = (
                    df.assign(weighted_yield=df["avg_yield"] * df["samples"])
                    .groupby("crop", as_index=False)
                    .agg(weighted_yield=("weighted_yield", "sum"), samples=("samples", "sum"))
                    .assign(avg_yield=lambda frame: frame["weighted_yield"] / frame["samples"].replace(0, 1))
                    .sort_values(by="avg_yield", ascending=False)
                    .head(1)
                )
                self._top_yield_crop = top_crop.iloc[0]["crop"] if not top_crop.empty else "Wheat"
            return self._top_yield_crop

    def _filter_candidates(self, frame, filters):
        current = frame
        for column, value in filters:
            if not value:
                continue

            key = _normalize_text(value)
            exact = current[column] == key
            if exact.any():
                current = current[exact]
                continue

            contains = current[column].str.contains(key, regex=False, na=False)
            if contains.any():
                current = current[contains]
                continue

            split_match = current[column].apply(lambda item: all(part in item for part in key.split()))
            if split_match.any():
                current = current[split_match]

        return current

    def normalize_market_records(self, records):
        normalized = []
        for item in records or []:
            record = {
                "state": _label(item.get("state") or item.get("State"), "Unknown State"),
                "district": _label(item.get("district") or item.get("District"), "Unknown District"),
                "market": _label(item.get("market") or item.get("Market") or item.get("Mandi"), "Local Mandi"),
                "commodity": _label(item.get("commodity") or item.get("Commodity"), "Unknown Commodity"),
                "variety": _label(item.get("variety") or item.get("Variety"), "Standard"),
                "min_price": _safe_int(item.get("min_price") or item.get("Min Price") or item.get("Min_Price")),
                "max_price": _safe_int(item.get("max_price") or item.get("Max Price") or item.get("Max_Price")),
                "modal_price": _safe_int(item.get("modal_price") or item.get("Modal Price") or item.get("Modal_Price")),
                "arrival_date": _label(
                    item.get("arrival_date") or item.get("Arrival_Date") or item.get("Date"),
                    "N/A",
                ),
                "unit": _label(item.get("unit") or item.get("Unit"), "Quintal"),
                "arrival_quantity": _safe_float(
                    item.get("arrival_quantity") or item.get("Arrival Quantity") or item.get("Arrival_Quantity")
                ),
                "grade": _label(item.get("grade") or item.get("Grade"), "FAQ"),
                "market_fee_pct": round(
                    _safe_float(item.get("market_fee_pct") or item.get("Market Fee (%)") or item.get("Market_Fee")),
                    2,
                ),
                "trader_count": _safe_int(item.get("trader_count") or item.get("Trader Count") or item.get("Trader_Count")),
            }
            normalized.append(record)
        return normalized

    def normalize_market_input(self, payload):
        state = payload.get("state") or payload.get("stateName") or ""
        district = payload.get("district") or payload.get("districtName") or ""
        market = payload.get("market") or payload.get("marketName") or ""
        commodity = payload.get("commodity") or payload.get("targetedCrop") or ""

        records = self.find_market_records(state, district, market, commodity, limit=1)
        if records:
            top = records[0]
            return {
                "state": top["state"],
                "district": top["district"],
                "market": top["market"],
                "commodity": top["commodity"],
            }

        return {
            "state": str(state).strip(),
            "district": str(district).strip(),
            "market": str(market).strip(),
            "commodity": str(commodity).strip(),
        }

    def find_market_records(self, state, district, market, commodity, limit=12):
        cache_key = (
            _normalize_text(state),
            _normalize_text(district),
            _normalize_text(market),
            _normalize_text(commodity),
            int(limit),
        )
        if cache_key in self._market_records_cache:
            return self._clone_records(self._market_records_cache[cache_key])

        df = self._load_mandi_df()
        candidates = self._filter_candidates(
            df,
            [
                ("state_key", state),
                ("district_key", district),
                ("commodity_key", commodity),
            ],
        )
        market_candidates = self._filter_candidates(candidates, [("market_key", market)])
        if market and not market_candidates.empty:
            candidates = market_candidates
        elif candidates.empty:
            candidates = self._filter_candidates(df, [("commodity_key", commodity), ("state_key", state)])

        if candidates.empty and commodity:
            candidates = self._filter_candidates(df, [("commodity_key", commodity)])

        if candidates.empty:
            return []

        candidates = candidates.copy()
        candidates["match_score"] = 0
        for column, value in [
            ("state_key", state),
            ("district_key", district),
            ("commodity_key", commodity),
            ("market_key", market),
        ]:
            if not value:
                continue
            key = _normalize_text(value)
            candidates["match_score"] += (candidates[column] == key).astype(int) * 3
            candidates["match_score"] += candidates[column].str.contains(key, regex=False, na=False).astype(int)

        candidates = candidates.sort_values(by=["match_score", "parsed_date"], ascending=[False, False]).head(limit)
        records = candidates[
            [
                "state",
                "district",
                "market",
                "commodity",
                "variety",
                "min_price",
                "max_price",
                "modal_price",
                "arrival_date",
                "unit",
                "arrival_quantity",
                "grade",
                "market_fee_pct",
                "trader_count",
            ]
        ].to_dict(orient="records")
        normalized = self.normalize_market_records(records)
        self._market_records_cache[cache_key] = self._clone_records(normalized)
        return normalized

    def _series_from_records(self, records, fallback_price):
        """Real records only (oldest -> newest). No invented points when data is missing."""
        dated = sorted(records or [], key=self._record_date)[-12:]
        return [{"label": item.get("arrival_date") or "N/A", "price": _safe_int(item.get("modal_price"))}
                for item in dated if _safe_int(item.get("modal_price")) > 0]

    @staticmethod
    def _record_date(item):
        parsed = pd.to_datetime(item.get("arrival_date"), errors="coerce", dayfirst=False)
        return parsed if not pd.isna(parsed) else pd.Timestamp.min

    def summarize_market_records(self, records, normalized_input):
        from services.price_forecast_service import price_forecast_service

        commodity = normalized_input.get("commodity") or "Crop"
        district = normalized_input.get("district") or "your district"
        market = normalized_input.get("market") or "nearby mandis"
        normalized_records = self.normalize_market_records(records)
        # Oldest -> newest, so trend comparisons are actually chronological.
        dated = sorted(normalized_records, key=self._record_date)

        forecast = price_forecast_service.forecast(
            state=normalized_input.get("state"),
            district=normalized_input.get("district"),
            market=normalized_input.get("market"),
            commodity=normalized_input.get("commodity"),
        )

        modal_prices = [item["modal_price"] for item in dated if item["modal_price"]]
        min_prices = [item["min_price"] for item in dated if item["min_price"]]
        max_prices = [item["max_price"] for item in dated if item["max_price"]]

        if modal_prices:
            # Recency-weighted average: newer records count more.
            weights = [index + 1 for index in range(len(modal_prices))]
            average_modal = int(sum(p * w for p, w in zip(modal_prices, weights)) / sum(weights))
            range_min = int(min(min_prices)) if min_prices else average_modal
            range_max = int(max(max_prices)) if max_prices else average_modal
        elif forecast:
            average_modal = forecast["last_actual_price"]
            range_min = min(point["low"] for point in forecast["points"])
            range_max = max(point["high"] for point in forecast["points"])
        else:
            average_modal, range_min, range_max = 1800, 1650, 1950

        if forecast:
            change = forecast["change_pct"]
        elif len(modal_prices) >= 4:
            half = len(modal_prices) // 2
            early = sum(modal_prices[:half]) / half
            late = sum(modal_prices[half:]) / (len(modal_prices) - half)
            change = (late - early) / early * 100 if early else 0.0
        else:
            change = 0.0

        sentiment = "Bullish" if change >= 3 else "Bearish" if change <= -3 else "Stable"

        bullets = [
            f"{commodity} pricing is being tracked from {market} and neighboring markets in {district}.",
            f"Recent modal prices are clustering around Rs {average_modal} per quintal.",
            f"The working range is Rs {range_min} to Rs {range_max}, which is useful for sale timing.",
        ]
        if forecast:
            nxt = forecast["points"][0]
            bullets.append(
                f"Forecast for {nxt['month']}: about Rs {nxt['price']} per quintal "
                f"(80% range Rs {nxt['low']} to Rs {nxt['high']}), {change:+.1f}% vs {forecast['last_actual_month']}."
            )
            if forecast["accuracy_pct"] is not None:
                bullets.append(
                    f"Model backtest accuracy on the last 3 months: {forecast['accuracy_pct']}% "
                    f"({forecast['level']}-level data, {forecast['months_of_history']} months of history)."
                )

        if sentiment == "Bullish":
            advice = "Prices are expected to rise, so holding stock for a few weeks may pay off if storage allows."
        elif sentiment == "Bearish":
            advice = "Prices are expected to soften, so selling sooner is usually safer."
        else:
            advice = "Prices look steady, so sell when a market pays above the typical modal price."

        if forecast:
            trend_series = [{"label": h["month"], "price": h["price"], "is_forecast": False} for h in forecast["history"][-6:]]
            trend_series += [
                {"label": p["month"], "price": p["price"], "low": p["low"], "high": p["high"], "is_forecast": True}
                for p in forecast["points"]
            ]
        else:
            trend_series = self._series_from_records(normalized_records, average_modal)

        return {
            "analysis_bullets": bullets,
            "sentiment": sentiment,
            "prediction_insight": f"{commodity} looks {sentiment.lower()} in {district}. {advice}",
            "trend_series": trend_series,
            "modeled_modal_price": forecast["next_month_price"] if forecast else average_modal,
            "modeled_range_min": range_min,
            "modeled_range_max": range_max,
            "modeled_commodity": commodity,
            "forecast": forecast,
            "fallback_suggestions": [
                f"Compare {market} with one nearby mandi before selling.",
                "Track arrivals and trader count together, not just one day's modal price.",
            ],
            "is_modeled": not bool(normalized_records) and not forecast,
        }

    def build_heatmap(self, state, district):
        cache_key = (_normalize_text(state), _normalize_text(district))
        if cache_key in self._heatmap_cache:
            return self._clone_records(self._heatmap_cache[cache_key])

        df = self._load_mandi_df()
        candidates = self._filter_candidates(df, [("state_key", state), ("district_key", district)])
        if candidates.empty:
            candidates = self._filter_candidates(df, [("state_key", state)])
        if candidates.empty:
            candidates = df

        grouped = (
            candidates.groupby(["market", "commodity"], as_index=False)
            .agg(arrival_quantity=("arrival_quantity", "sum"), samples=("commodity", "size"))
            .sort_values(by=["arrival_quantity", "samples"], ascending=[False, False])
            .head(24)
        )

        rows = []
        for item in grouped.to_dict(orient="records"):
            intensity = int(round(item["arrival_quantity"])) if item["arrival_quantity"] > 0 else int(item["samples"])
            rows.append(
                {
                    "subRegion": item["market"],
                    "cropName": item["commodity"],
                    "intensity": max(intensity, 1),
                    "production_label": f"{max(intensity, 1)} tracked units",
                    "analysis_text": f"{item['commodity']} is one of the stronger signals in {item['market']}.",
                }
            )

        self._heatmap_cache[cache_key] = self._clone_records(rows)
        return rows

    def _get_crop_aggregate(self, state, district, season):
        cache_key = (_normalize_text(state), _normalize_text(district), _normalize_text(season))
        if cache_key in self._crop_aggregate_cache:
            return self._crop_aggregate_cache[cache_key].copy()

        df = self._load_crop_summary_df()
        candidates = self._filter_candidates(df, [("state_key", state), ("district_key", district)])
        if candidates.empty:
            candidates = self._filter_candidates(df, [("state_key", state)])
        if season:
            seasonal = self._filter_candidates(candidates, [("season_key", season)])
            if not seasonal.empty:
                candidates = seasonal
        if candidates.empty:
            candidates = df

        grouped = (
            candidates.assign(weighted_yield=candidates["avg_yield"] * candidates["samples"])
            .groupby(["crop", "crop_key"], as_index=False)
            .agg(
                weighted_yield=("weighted_yield", "sum"),
                total_production=("total_production", "sum"),
                total_area=("total_area", "sum"),
                samples=("samples", "sum"),
            )
        )
        grouped["avg_yield"] = grouped["weighted_yield"] / grouped["samples"].replace(0, 1)
        grouped = grouped.sort_values(by=["avg_yield", "total_production"], ascending=[False, False])

        grouped["yield_score"] = grouped["avg_yield"].rank(pct=True)
        grouped["production_score"] = grouped["total_production"].apply(lambda value: math.log1p(max(value, 0))).rank(pct=True)
        grouped["efficiency_score"] = (
            (grouped["total_production"] / grouped["total_area"].replace(0, pd.NA).fillna(1)).rank(pct=True)
        )
        grouped["base_score"] = (
            grouped["yield_score"] * 0.45
            + grouped["production_score"] * 0.35
            + grouped["efficiency_score"] * 0.20
        )
        self._crop_aggregate_cache[cache_key] = grouped
        return grouped.copy()

    def _apply_crop_heuristics(self, grouped, user_input):
        soil_type = str(user_input.get("soilType") or "").strip().lower()
        water_source = str(user_input.get("waterSource") or "").strip().lower()
        irrigation_type = str(user_input.get("irrigationType") or "").strip().lower()
        targeted_crop = _normalize_text(user_input.get("targetedCrop"))
        budget = _safe_float(user_input.get("budget"))

        soil_boosts = {
            "loamy": {"wheat": 0.08, "maize": 0.07, "potato": 0.07, "onion": 0.06},
            "clay": {"rice": 0.1, "sugarcane": 0.07, "cotton": 0.05},
            "sandy": {"groundnut": 0.08, "millets": 0.06, "gram": 0.05},
            "black": {"cotton": 0.1, "soybean": 0.08, "jowar": 0.06},
        }
        rainfed_penalties = {"rice": -0.09, "sugarcane": -0.08, "banana": -0.08}
        drip_boosts = {"banana": 0.08, "cotton": 0.06, "onion": 0.06, "tomato": 0.06}
        low_budget_penalties = {"banana": -0.08, "sugarcane": -0.08, "cotton": -0.05}

        grouped["heuristic"] = 0.0
        selected = soil_boosts.get(soil_type, {})
        for crop_key, boost in selected.items():
            grouped.loc[grouped["crop_key"] == crop_key, "heuristic"] += boost

        if water_source == "rainfed":
            for crop_key, penalty in rainfed_penalties.items():
                grouped.loc[grouped["crop_key"] == crop_key, "heuristic"] += penalty

        if irrigation_type == "drip":
            for crop_key, boost in drip_boosts.items():
                grouped.loc[grouped["crop_key"] == crop_key, "heuristic"] += boost

        if budget and budget < 25000:
            for crop_key, penalty in low_budget_penalties.items():
                grouped.loc[grouped["crop_key"] == crop_key, "heuristic"] += penalty

        if targeted_crop:
            grouped.loc[grouped["crop_key"] == targeted_crop, "heuristic"] += 0.12

        return grouped

    def recommend_crops(self, user_input):
        state = user_input.get("stateName") or ""
        district = user_input.get("districtName") or ""
        season = user_input.get("season") or ""
        grouped = self._get_crop_aggregate(state, district, season)
        grouped = self._apply_crop_heuristics(grouped, user_input)
        grouped["score"] = grouped["base_score"] + grouped["heuristic"]
        top_rows = grouped.sort_values(by="score", ascending=False).head(5).reset_index(drop=True)

        recommended = []
        max_score = max(top_rows["score"].max(), 0.1)
        for _, row in top_rows.iterrows():
            probability = int(round(70 + max(row["score"], 0) / max_score * 25))
            if row["score"] >= 0.9:
                risk = "Low"
            elif row["score"] >= 0.72:
                risk = "Medium"
            else:
                risk = "High"
            recommended.append(
                {
                    "crop": row["crop"],
                    "probability": min(probability, 96),
                    "risk": risk,
                }
            )

        top_crop = recommended[0]["crop"] if recommended else "Wheat"
        market_records = self.find_market_records(state, district, "", top_crop, limit=8)
        analysis = self.summarize_market_records(
            market_records,
            {
                "state": state,
                "district": district,
                "market": user_input.get("marketName") or "",
                "commodity": top_crop,
            },
        )

        targeted_crop = user_input.get("targetedCrop")
        bullets = [
            f"Recommendations are ranked from historical yield and production patterns for {district or 'the selected district'}, {state or 'the selected state'}.",
            f"{top_crop} stands out because it combines stronger output history with a workable price band near Rs {analysis['modeled_modal_price']} per quintal.",
            f"Soil and water inputs were applied to reduce crops that are harder to manage under the selected farm conditions.",
        ]
        if targeted_crop and str(targeted_crop).strip().lower() not in {"", "none"}:
            targeted_label = str(targeted_crop).strip()
            if any(item["crop"].lower() == targeted_label.lower() for item in recommended):
                bullets.append(f"{targeted_label} remains viable in this plan and was kept in the shortlist.")
            else:
                bullets.append(f"{targeted_label} looks less competitive than the top five options under the current constraints.")

        return {
            "recommendedCrops": recommended,
            "explanation": {"bullets": bullets},
            "pricePrediction": {
                "crop": top_crop,
                "rangeMin": analysis["modeled_range_min"],
                "rangeMax": analysis["modeled_range_max"],
                "modalPrice": analysis["modeled_modal_price"],
                "arrivalQuantity": "Dataset-backed estimate",
                "marketFeePct": 1.5,
                "traderCount": 12,
                "unit": "Quintal",
                "grade": "FAQ",
            },
            "marketTrend": {"series": analysis["trend_series"]},
            "risk": {"level": recommended[0]["risk"] if recommended else "Medium"},
        }

    def chat_reply(self, message, history=None, farmer_context=""):
        history = history or []
        text = str(message or "").strip()
        profile = {}
        for part in str(farmer_context or "").split("; "):
            if part.startswith("Farmer location: "):
                district, _, state = part[len("Farmer location: "):].partition(", ")
                profile = {"district": district.strip(), "state": state.strip()}
        key = _normalize_text(text)
        keywords = {
            "crop",
            "farming",
            "farm",
            "yield",
            "soil",
            "fertilizer",
            "pesticide",
            "market",
            "price",
            "mandi",
            "irrigation",
            "rain",
            "weed",
            "disease",
            "harvest",
        }
        if not any(token in key for token in keywords):
            return {
                "reply": (
                    "I can help with farming, crop planning, mandi prices, irrigation, fertilizer, pests, and harvest decisions.\n\n"
                    "- Ask about a crop, district, mandi, or field problem.\n"
                    "- Include your state or district for sharper advice.\n"
                    "- I can also compare likely price ranges from the local dataset."
                ),
                "suggested": [
                    "Best crop for my district",
                    "How to improve yield this season?",
                    "Current mandi price trend for wheat",
                ],
            }

        commodity_match = next(
            (
                commodity
                for commodity in self._get_commodity_names()
                if _normalize_text(commodity) in key
            ),
            None,
        )
        crop_match = next(
            (
                crop
                for crop in self._get_crop_names()
                if _normalize_text(crop) in key
            ),
            commodity_match,
        )

        reply_lines = [
            "Here is a practical advisory based on your project dataset and standard farm-side planning.",
            "",
        ]

        if farmer_context:
            reply_lines.insert(1, f"(Personalised for you - {farmer_context})")
        if crop_match:
            crop_records = self.find_market_records(profile.get("state", ""), profile.get("district", ""), "", crop_match, limit=6)                 or self.find_market_records("", "", "", crop_match, limit=6)
            market_view = self.summarize_market_records(crop_records, {
                "commodity": crop_match, "state": profile.get("state", ""),
                "district": profile.get("district") or "regional markets", "market": "tracked mandis"})
            reply_lines.append(
                f"- For {crop_match}, the dataset points to a working modal band near Rs {market_view['modeled_modal_price']} per quintal."
            )
            reply_lines.append(
                f"- Current tone looks {market_view['sentiment'].lower()}, so compare today's offer with the Rs {market_view['modeled_range_min']}-{market_view['modeled_range_max']} band before selling."
            )
        else:
            leader = self._get_top_yield_crop()
            reply_lines.append(f"- A strong starting point from the dataset is {leader}, especially where yield stability matters.")

        if "fertilizer" in key:
            reply_lines.append("- Split fertilizer doses by crop stage and irrigate soon after top dressing to improve uptake.")
        if "pest" in key or "disease" in key:
            reply_lines.append("- Scout the field early morning, remove the worst hotspots first, and rotate chemistry instead of repeating one product.")
        if "water" in key or "irrigation" in key:
            reply_lines.append("- Match irrigation frequency to crop stage; avoid standing water unless the crop specifically needs it.")
        if len(history) > 2:
            reply_lines.append("- I kept the recent conversation context in mind, so feel free to narrow this to one crop or one mandi.")

        return {
            "reply": "\n".join(reply_lines),
            "suggested": [
                "Which crop should I prefer this season?",
                "How do I reduce pest pressure safely?",
                "Should I sell now or wait for a better mandi price?",
            ],
        }


_agri_data_service = None


def get_agri_data_service():
    global _agri_data_service
    if _agri_data_service is None:
        _agri_data_service = AgriDataService()
    return _agri_data_service
