"""Train the soil + climate crop recommendation model.

Run:  python -m ml.train_soil_model        (from the backend folder)

data:     data/Crop_recommendation.csv (2,200 rows, 22 crops, 100 rows each)
features: N, P, K (soil nutrients, kg/ha), temperature (C), humidity (%), ph, rainfall (mm)
target:   crop label
model:    Random Forest classifier (300 trees) - simple, accurate on this data,
          and it gives a probability for every crop, so we can rank them.
evaluation: stratified 80/20 split + 5-fold cross-validation.

Saves ml/artifacts/soil_crop_model.joblib and adds "soil_model" to metrics.json.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

BACKEND = Path(__file__).resolve().parent.parent
DATA = BACKEND / "data" / "Crop_recommendation.csv"
ART = BACKEND / "ml" / "artifacts"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train():
    t = time.time()
    df = pd.read_csv(DATA).dropna()
    X, y = df[FEATURES], df["label"].str.strip().str.lower()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), X, y, cv=5)

    # Final model on all rows
    model.fit(X, y)
    ranges = {crop: {f: [round(float(g[f].quantile(0.1)), 1), round(float(g[f].quantile(0.9)), 1)] for f in FEATURES}
              for crop, g in df.assign(label=y).groupby("label")}

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "ranges": ranges}, ART / "soil_crop_model.joblib")

    metrics = {
        "rows": int(len(df)), "crops": int(y.nunique()),
        "split": "stratified 80/20 + 5-fold cross-validation",
        "test_accuracy_pct": round(accuracy_score(y_test, pred) * 100, 2),
        "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "cv_accuracy_pct": round(cv.mean() * 100, 2),
        "feature_importance": dict(sorted(
            ((f, round(float(v), 3)) for f, v in zip(FEATURES, model.feature_importances_)),
            key=lambda kv: -kv[1])),
        "model": "RandomForestClassifier(n_estimators=300)",
    }
    path = ART / "metrics.json"
    all_metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_metrics["soil_model"] = metrics
    path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Soil model trained in {time.time() - t:.1f}s: {json.dumps(metrics, indent=2)}")
    return metrics


if __name__ == "__main__":
    sys.exit(0 if train() else 1)
