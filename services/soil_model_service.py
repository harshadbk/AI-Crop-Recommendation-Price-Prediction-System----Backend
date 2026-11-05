"""Soil + climate crop recommendation (Random Forest on Crop_recommendation.csv).

Inputs come from three places, and every value is tagged with its source so the
farmer (and the examiner) can see where it came from:
  * N, P, K, pH         -> the farmer's Soil Health Card (asked once at registration),
                           otherwise a typical value for their soil type
  * temperature, humidity -> live OpenWeatherMap 5-day outlook for their district
  * rainfall            -> blend of the live forecast and the season's normal rainfall
"""

import threading

import pandas as pd

from core.config import settings

# Typical topsoil values (kg/ha, pH) by soil type, used when no soil test is available.
SOIL_DEFAULTS = {
    "black": dict(N=60, P=45, K=60, ph=7.8),
    "red": dict(N=35, P=30, K=30, ph=6.2),
    "alluvial": dict(N=70, P=50, K=45, ph=7.0),
    "laterite": dict(N=30, P=25, K=25, ph=5.5),
    "loamy": dict(N=60, P=50, K=45, ph=6.7),
    "clay": dict(N=55, P=40, K=50, ph=7.2),
    "sandy": dict(N=25, P=25, K=20, ph=6.8),
    "silt": dict(N=55, P=45, K=40, ph=6.8),
}
# Season normals when live weather is unavailable: (temp C, humidity %, rainfall mm)
SEASON_CLIMATE = {"rainy": (27, 80, 180), "winter": (21, 60, 70), "summer": (32, 45, 40)}

# Dataset label -> display name and the key used by the yield model (ml_service.CROP_INFO)
LABELS = {
    "rice": ("Rice (Paddy)", "rice"), "maize": ("Maize", "maize"), "chickpea": ("Chickpea (Gram)", "gram"),
    "kidneybeans": ("Kidney Beans (Rajma)", None), "pigeonpeas": ("Tur (Pigeon Pea)", "arhar tur"),
    "mothbeans": ("Moth Beans", None), "mungbean": ("Moong (Green Gram)", "moong green gram"),
    "blackgram": ("Urad (Black Gram)", "urad"), "lentil": ("Lentil (Masoor)", "masoor"),
    "pomegranate": ("Pomegranate", None), "banana": ("Banana", "banana"), "mango": ("Mango", None),
    "grapes": ("Grapes", None), "watermelon": ("Watermelon", None), "muskmelon": ("Muskmelon", None),
    "apple": ("Apple", None), "orange": ("Orange", None), "papaya": ("Papaya", None),
    "coconut": ("Coconut", None), "cotton": ("Cotton", "cotton lint"), "jute": ("Jute", "jute"),
    "coffee": ("Coffee", None),
}


# Crops in the dataset that only make sense in certain states (hill / plantation / delta crops).
REGIONAL_ONLY = {
    "apple": {"jammu and kashmir", "himachal pradesh", "uttarakhand", "arunachal pradesh"},
    "coffee": {"karnataka", "kerala", "tamil nadu"},
    "coconut": {"kerala", "tamil nadu", "karnataka", "andhra pradesh", "goa", "odisha", "west bengal"},
    "jute": {"west bengal", "bihar", "assam", "odisha", "meghalaya"},
    "kidneybeans": {"jammu and kashmir", "himachal pradesh", "uttarakhand"},
    "orange": {"maharashtra", "madhya pradesh", "rajasthan", "punjab", "assam", "meghalaya", "nagaland", "karnataka"},
}


def soil_family(soil_type):
    text = str(soil_type or "").lower()
    for family in SOIL_DEFAULTS:
        if family in text:
            return family
    if "regur" in text or "cotton" in text:
        return "black"
    return None


def _num(value):
    try:
        number = float(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


class SoilModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self.pack = None
        self.error = None

    def load(self):
        if self.pack is not None:
            return True
        with self._lock:
            if self.pack is None:
                try:
                    import joblib
                    self.pack = joblib.load(settings.ML_DIR / "soil_crop_model.joblib")
                except Exception as exc:
                    self.error = str(exc)
                    print(f"Soil model unavailable (run: python -m ml.train_soil_model): {exc}")
        return self.pack is not None

    def build_inputs(self, payload, weather=None, season_key="winter"):
        """Collect the 7 model inputs and remember where each came from."""
        values, sources = {}, {}
        family = soil_family(payload.get("soilType"))
        defaults = SOIL_DEFAULTS.get(family, SOIL_DEFAULTS["loamy"])
        for name, field in (("N", "soilN"), ("P", "soilP"), ("K", "soilK"), ("ph", "soilPh")):
            given = _num(payload.get(field))
            if name == "ph" and given is not None and not (3 <= given <= 10):
                given = None
            if given is not None:
                values[name], sources[name] = given, "your soil test"
            else:
                values[name] = defaults[name]
                sources[name] = f"typical for {family or 'loamy'} soil"

        temp_n, hum_n, rain_n = SEASON_CLIMATE.get(season_key, SEASON_CLIMATE["winter"])
        outlook = (weather or {}).get("outlook") or {}
        if outlook.get("temp_avg") is not None:
            values["temperature"], sources["temperature"] = float(outlook["temp_avg"]), "live weather (5-day avg)"
        else:
            values["temperature"], sources["temperature"] = temp_n, "season normal"
        if outlook.get("avg_humidity") is not None:
            values["humidity"], sources["humidity"] = float(outlook["avg_humidity"]), "live weather (5-day avg)"
        else:
            values["humidity"], sources["humidity"] = hum_n, "season normal"
        if outlook.get("total_rain_mm") is not None:
            # 5-day forecast scaled to a month, blended with the season normal
            monthly = float(outlook["total_rain_mm"]) * 6
            values["rainfall"] = round(min(300.0, max(20.0, 0.5 * rain_n + 0.5 * monthly)), 1)
            sources["rainfall"] = "live forecast + season normal"
        else:
            values["rainfall"], sources["rainfall"] = rain_n, "season normal"
        return values, sources

    def predict(self, payload, weather=None, season_key="winter", top_n=5):
        """Top crops with probability (%) from the Random Forest model, or None."""
        if not self.load():
            return None
        values, sources = self.build_inputs(payload, weather, season_key)
        frame = pd.DataFrame([[values[f] for f in self.pack["features"]]], columns=self.pack["features"])
        probs = self.pack["model"].predict_proba(frame)[0]
        state = str(payload.get("stateName") or "").strip().lower()
        ranked = [(label, prob) for label, prob in zip(self.pack["model"].classes_, probs)
                  if not state or label not in REGIONAL_ONLY or state in REGIONAL_ONLY[label]]
        total = sum(p for _, p in ranked) or 1.0
        ranked = sorted(((label, prob / total) for label, prob in ranked), key=lambda kv: -kv[1])

        crops = []
        for label, prob in ranked[:top_n]:
            name, ml_key = LABELS.get(label, (label.title(), None))
            crops.append({"crop": name, "label": label, "ml_key": ml_key,
                          "probability": round(float(prob) * 100, 1),
                          "ideal": self.pack["ranges"].get(label)})
        return {
            "crops": crops,
            "all_probabilities": {LABELS.get(l, (l, None))[1] or l: float(p) for l, p in ranked},
            "inputs": {k: round(v, 1) for k, v in values.items()},
            "input_sources": sources,
            "soil_test_used": any(s == "your soil test" for s in sources.values()),
            "model": "Random Forest (Crop_recommendation.csv)",
        }


soil_model_service = SoilModelService()
