"""Crop calendar: what a farmer should be doing this month for each of their crops.

Each crop has one or more growing windows (sowing months -> harvest months).
For the current month we work out the crop stage (land preparation, sowing,
early / mid / late growth, harvest) and return simple, actionable tasks. The
weather summary from OpenWeatherMap is then used to adjust the advice
(e.g. "rain expected - postpone spraying").

Windows follow common practice in Maharashtra / central India; they are a
planning guide, not a substitute for local KVK advice.
"""

import re
from datetime import datetime

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# sow / harvest are month numbers (1-12). "pests" and "critical" are crop-specific tips.
CALENDAR = {
    "onion": dict(name="Onion", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[10, 11]),
        dict(season="Late Kharif", sow=[8, 9], harvest=[1, 2]),
        dict(season="Rabi", sow=[10, 11, 12], harvest=[3, 4, 5]),
    ], pests="thrips and purple blotch", critical="bulb formation (about 60-90 days after transplanting)",
        tip="Stop irrigation 10-15 days before harvest and cure bulbs in shade for better storage."),
    "grapes": dict(name="Grapes", perennial={
        4: ["Back (foundation) pruning after harvest", "Apply FYM and basal fertiliser", "Train new canes on the trellis"],
        5: ["Keep canes growing; irrigate regularly in heat", "Spray for flea beetle if seen"],
        6: ["Fruitful cane development: limit nitrogen", "Ensure drainage before monsoon"],
        7: ["Monsoon: watch for downy mildew; spray only in dry spells", "Remove extra shoots"],
        8: ["Downy mildew and anthracnose control in wet weather", "Cane maturity: reduce irrigation"],
        9: ["Prepare for forward (fruit) pruning", "Soil test and plan fertiliser"],
        10: ["Forward (fruit) pruning and hydrogen cyanamide paste for bud break", "Downy mildew protection after pruning"],
        11: ["Shoot thinning; GA3 sprays for bunch elongation as advised", "Powdery mildew watch"],
        12: ["Berry thinning and dipping", "Irrigate as per berry growth; avoid water stress"],
        1: ["Berry development: potash feeding", "Watch powdery mildew and mealybug"],
        2: ["Harvest begins for early-pruned gardens; check export/ mandi rates", "Stop sprays before harvest (PHI)"],
        3: ["Main harvest; grade and pack carefully", "Plan back pruning after harvest"],
    }, pests="downy mildew, powdery mildew and mealybug"),
    "pomegranate": dict(name="Pomegranate", perennial={
        m: tasks for m, tasks in {
            1: ["Ambe bahar: give rest and stop irrigation", "Prune dead and crossing branches"],
            2: ["Ambe bahar flowering: resume light irrigation", "Bacterial blight spray schedule"],
            5: ["Mrig bahar: give rest (stop water)", "Clean orchard, remove infected fruits"],
            6: ["Mrig bahar: resume irrigation after rains; apply fertiliser", "Bacterial blight watch in humid weather"],
            7: ["Flowering and fruit set; avoid water stress", "Fruit borer: bag fruits if possible"],
            9: ["Hasta bahar: give rest", "Copper spray for blight"],
            10: ["Hasta bahar: resume irrigation and fertiliser", "Remove cracked fruits"],
        }.items()
    }, pests="bacterial blight (oily spot) and fruit borer"),
    "sugarcane": dict(name="Sugarcane", windows=[
        dict(season="Adsali", sow=[7, 8], harvest=[10, 11, 12]),
        dict(season="Pre-seasonal", sow=[10, 11], harvest=[12, 1, 2]),
        dict(season="Suru", sow=[1, 2], harvest=[12, 1, 2, 3]),
    ], pests="woolly aphid and top shoot borer", critical="grand growth phase (4-9 months)",
        tip="Earthing up at 3-4 months prevents lodging; trash mulching saves water."),
    "wheat": dict(name="Wheat", windows=[dict(season="Rabi", sow=[11, 12], harvest=[3, 4])],
                  pests="aphids and rust", critical="crown root initiation (21 days after sowing)",
                  tip="Give the first irrigation at 21 days (CRI stage) - it has the biggest effect on yield."),
    "chickpea": dict(name="Chickpea (Gram)", windows=[dict(season="Rabi", sow=[10, 11], harvest=[2, 3])],
                     pests="pod borer (Helicoverpa) and wilt", critical="flowering and pod filling",
                     tip="Nipping (removing top buds) at 30-40 days increases branches and pods."),
    "soybean": dict(name="Soybean", windows=[dict(season="Kharif", sow=[6, 7], harvest=[9, 10])],
                    pests="girdle beetle, stem fly and leaf-eating caterpillars", critical="flowering and pod filling",
                    tip="Sow only after 75-100 mm rain; use broad bed furrow (BBF) for drainage."),
    "cotton": dict(name="Cotton", windows=[dict(season="Kharif", sow=[5, 6, 7], harvest=[10, 11, 12, 1])],
                   pests="pink bollworm, whitefly and sucking pests", critical="square formation and boll development",
                   tip="Use pheromone traps (5/acre) for pink bollworm and avoid extending the crop beyond January."),
    "tur": dict(name="Tur (Pigeon Pea)", windows=[dict(season="Kharif", sow=[6, 7], harvest=[12, 1, 2])],
                pests="pod borer and pod fly", critical="flowering and pod formation",
                tip="Intercrop with soybean or cotton for better income per acre."),
    "maize": dict(name="Maize", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[9, 10]),
        dict(season="Rabi", sow=[10, 11], harvest=[2, 3]),
    ], pests="fall armyworm", critical="tasselling and silking",
        tip="Check the leaf whorls weekly for fall armyworm in the first 30 days."),
    "rice": dict(name="Rice (Paddy)", windows=[dict(season="Kharif", sow=[6, 7], harvest=[10, 11])],
                 pests="stem borer, blast and brown plant hopper", critical="tillering and flowering",
                 tip="Keep 2-5 cm standing water during tillering; drain 10 days before harvest."),
    "tomato": dict(name="Tomato", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[9, 10, 11]),
        dict(season="Rabi", sow=[10, 11], harvest=[1, 2, 3]),
        dict(season="Summer", sow=[1, 2], harvest=[4, 5]),
    ], pests="fruit borer, leaf curl virus (whitefly) and early blight", critical="flowering and fruit set",
        tip="Staking and drip with mulch give cleaner fruit and fewer diseases."),
    "potato": dict(name="Potato", windows=[dict(season="Rabi", sow=[10, 11], harvest=[1, 2, 3])],
                   pests="late blight and aphids", critical="tuber initiation and bulking",
                   tip="Earth up at 25-30 days; spray for late blight before foggy, humid spells."),
    "groundnut": dict(name="Groundnut", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[10, 11]),
        dict(season="Summer", sow=[1, 2], harvest=[5, 6]),
    ], pests="leaf spot, white grub and leaf miner", critical="pegging and pod development",
        tip="Apply gypsum at flowering for better pod filling."),
    "mustard": dict(name="Mustard", windows=[dict(season="Rabi", sow=[10, 11], harvest=[2, 3])],
                    pests="aphids", critical="flowering and siliqua formation",
                    tip="Thin plants to 10-15 cm spacing at 15-20 days."),
    "jowar": dict(name="Jowar (Sorghum)", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[10, 11]),
        dict(season="Rabi", sow=[9, 10], harvest=[1, 2]),
    ], pests="shoot fly and stem borer", critical="booting and grain filling",
        tip="Rabi jowar does well on black soil with stored moisture; sow early October."),
    "bajra": dict(name="Bajra (Pearl Millet)", windows=[dict(season="Kharif", sow=[6, 7], harvest=[9, 10])],
                  pests="downy mildew (green ear)", critical="flowering",
                  tip="Drought tolerant; good choice for light soils and low rainfall."),
    "chilli": dict(name="Chilli", windows=[
        dict(season="Kharif", sow=[6, 7], harvest=[10, 11, 12, 1]),
        dict(season="Rabi", sow=[10, 11], harvest=[2, 3, 4]),
    ], pests="thrips, mites and leaf curl", critical="flowering and fruiting",
        tip="Use yellow and blue sticky traps to monitor sucking pests."),
    "banana": dict(name="Banana", windows=[dict(season="Mrig bahar", sow=[6, 7], harvest=[6, 7, 8])],
                   pests="sigatoka leaf spot and bunchy top", critical="bunch emergence",
                   tip="De-sucker regularly and prop plants before bunch emergence."),
}

ALIASES = {
    "chana": "chickpea", "gram": "chickpea", "bengal gram": "chickpea", "chickpea chana": "chickpea",
    "pigeon pea": "tur", "arhar": "tur", "tur dal": "tur", "pigeon pea tur": "tur", "arhar tur": "tur", "red gram": "tur",
    "soyabean": "soybean", "soya": "soybean", "paddy": "rice", "dhan": "rice",
    "green chilli": "chilli", "chili": "chilli", "dry chillies": "chilli", "mirchi": "chilli",
    "kanda": "onion", "draksh": "grapes", "grape": "grapes", "anar": "pomegranate", "dalimb": "pomegranate",
    "sorghum": "jowar", "pearl millet": "bajra", "rapeseed mustard": "mustard", "cotton lint": "cotton",
    "kapus": "cotton", "ganna": "sugarcane", "oos": "sugarcane", "makka": "maize", "corn": "maize",
    "gehu": "wheat", "tamatar": "tomato", "aloo": "potato", "moongphali": "groundnut",
}

STAGE_TASKS = {
    "land_prep": ["Plough and level the field; add 4-5 tonnes FYM / compost per acre",
                  "Get a soil test done and buy certified seed in advance"],
    "sowing": ["Sowing window is open: use certified seed and seed treatment (Trichoderma / fungicide)",
               "Apply basal dose of fertiliser as per soil test"],
    "early": ["Gap filling and first weeding", "First top dressing of nitrogen"],
    "mid": ["Second top dressing; irrigate at the critical stage", "Scout the field twice a week for pests"],
    "late": ["Stop nitrogen; keep monitoring pests and disease", "Plan harvest labour and check mandi prices"],
    "harvest": ["Harvest at proper maturity; dry and grade the produce",
                "Compare mandi prices before selling; store if prices are low"],
}
STAGE_LABEL = {"land_prep": "Land preparation", "sowing": "Sowing", "early": "Early growth",
               "mid": "Mid growth", "late": "Maturity", "harvest": "Harvest", "perennial": "Orchard care",
               "off": "Off season"}


def crop_key(name):
    text = re.sub(r"[^a-z ]+", " ", str(name or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    if text in CALENDAR:
        return text
    if text in ALIASES:
        return ALIASES[text]
    for word in text.split():
        if word in CALENDAR:
            return word
        if word in ALIASES:
            return ALIASES[word]
    return None


def _months_between(start, end):
    """Months from start to end going forward around the year (0..11)."""
    return (end - start) % 12


def _stage(window, month):
    sow, harvest = window["sow"], window["harvest"]
    if month in harvest:
        return "harvest"
    if month in sow:
        return "sowing"
    if (sow[0] - month) % 12 == 1:
        return "land_prep"
    grow_months = _months_between(sow[-1], harvest[0])
    since = _months_between(sow[-1], month)
    if 0 < since < grow_months:
        ratio = since / grow_months
        return "early" if ratio <= 0.34 else "mid" if ratio <= 0.67 else "late"
    return None


def crop_activities(crop_name, month=None):
    """Stage + tasks for one crop this month, or None if the crop is unknown."""
    month = month or datetime.now().month
    key = crop_key(crop_name)
    if not key:
        return None
    info = CALENDAR[key]

    if "perennial" in info:
        tasks = info["perennial"].get(month) or ["Regular orchard care: irrigation, nutrition and pest scouting"]
        return {"crop": info["name"], "key": key, "stage": "perennial", "stage_label": STAGE_LABEL["perennial"],
                "season": "Perennial", "tasks": list(tasks), "watch_for": info.get("pests")}

    # pick the most relevant window (active stage beats off-season)
    order = ["harvest", "sowing", "mid", "late", "early", "land_prep"]
    best = None
    for window in info["windows"]:
        stage = _stage(window, month)
        if stage and (best is None or order.index(stage) < order.index(best[1])):
            best = (window, stage)

    if not best:
        upcoming = min(info["windows"], key=lambda w: _months_between(month, w["sow"][0]))
        return {"crop": info["name"], "key": key, "stage": "off", "stage_label": STAGE_LABEL["off"],
                "season": upcoming["season"],
                "tasks": [f"Next sowing window: {upcoming['season']} ({MONTHS[upcoming['sow'][0] - 1]}). "
                          "Use this time for soil testing, green manure or a short catch crop."],
                "watch_for": info.get("pests")}

    window, stage = best
    tasks = list(STAGE_TASKS[stage])
    if stage in ("mid", "late") and info.get("pests"):
        tasks.append(f"Main threats now: {info['pests']}")
    if stage in ("early", "mid") and info.get("critical"):
        tasks.append(f"Do not miss water at {info['critical']}")
    if stage in ("sowing", "early") and info.get("tip"):
        tasks.append(info["tip"])
    return {"crop": info["name"], "key": key, "stage": stage, "stage_label": STAGE_LABEL[stage],
            "season": window["season"],
            "sow_months": [MONTHS[m - 1] for m in window["sow"]],
            "harvest_months": [MONTHS[m - 1] for m in window["harvest"]],
            "tasks": tasks, "watch_for": info.get("pests")}


def weather_actions(weather):
    """Turn the weather summary into farm actions for this week."""
    if not weather:
        return []
    actions = list(weather.get("alerts") or [])
    outlook = weather.get("outlook") or {}
    spray = weather.get("good_spray_days") or []
    if spray:
        actions.append(f"Best days for spraying (low rain and wind): {', '.join(spray)}")
    elif outlook:
        actions.append("No good spraying day this week - rain or wind expected. Postpone sprays.")
    rain = outlook.get("total_rain_mm")
    if rain is not None:
        if rain >= 25:
            actions.append(f"About {rain} mm rain expected in {outlook.get('days', 5)} days - skip irrigation and keep drains open.")
        elif rain < 2:
            actions.append("Little or no rain expected - plan irrigation for crops at critical stages.")
    return actions
