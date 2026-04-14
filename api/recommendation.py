import json
import traceback
from fastapi import APIRouter, HTTPException
from backend.models.request_models import RecommendationRequest
from backend.services.rag_service import get_rag_service

router = APIRouter(prefix="/api")

@router.post("/crop-prediction")
async def get_live_recommendation(req: RecommendationRequest):
    """RAG-based AI Recommendation"""
    try:
        # Get the service lazily
        rag = get_rag_service()
        raw_result = await rag.generate_recommendation(req.dict())
        
        # Clean up JSON if LLM added markdown blockers
        clean_json = raw_result.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        result_data = json.loads(clean_json)
        result_data["mode"] = "live_rag"
        return result_data

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"RAG Error Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"AI RAG Error: {str(e)}")
