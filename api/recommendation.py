import json

from fastapi import APIRouter, Depends, HTTPException

from core.config import settings
from core.security import get_optional_user
from models.request_models import RecommendationRequest
from services import personalization_service as personal
from services.agri_data_service import get_agri_data_service
from services.rag_service import get_rag_service

router = APIRouter(prefix="/api")


def as_payload(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def get_ready_rag_service():
    if not settings.RAG_ENABLED:
        return None
    rag = get_rag_service()
    return rag if rag.available else None


async def _rag_recommendation(payload):
    rag = get_ready_rag_service()
    if rag is None:
        return None
    try:
        raw = (await rag.generate_recommendation(payload)).strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        result = json.loads(raw)
        result["mode"] = "live_rag"
        return result
    except Exception as exc:
        print(f"RAG recommendation unavailable: {exc}")
        return None


@router.post("/crop-prediction")
async def get_live_recommendation(req: RecommendationRequest, user=Depends(get_optional_user)):
    """Personalised crop recommendation.

    1. Any field the farmer leaves empty is filled from their saved profile.
    2. ML path (default): soil Random Forest + XGBoost yield model + live weather.
    3. Fallbacks: RAG (Groq + Weaviate) if the ML models are missing, then dataset scoring.
    """
    payload = personal.merge_profile(as_payload(req), user)
    if not payload.get("stateName") or not payload.get("districtName"):
        raise HTTPException(status_code=400, detail="Please choose a state and district (or log in to use your farm profile).")

    result = await personal.recommend(payload)
    if result.get("mode") == "ml_live_weather":
        result["personalized"] = bool(user)
        return result

    rag_result = await _rag_recommendation(payload)
    if rag_result:
        rag_result.setdefault("weather", result.get("weather"))
        rag_result.setdefault("soilModel", result.get("soilModel"))
        rag_result["personalized"] = bool(user)
        return rag_result

    if not result.get("recommendedCrops"):
        result = get_agri_data_service().recommend_crops(payload)
        result["mode"] = "offline_dataset"
    result["personalized"] = bool(user)
    return result
