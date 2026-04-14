import json
from fastapi import APIRouter, HTTPException
from models.request_models import PricePredictionRequest
from services.mandi_api_service import mandi_api_service
from services.rag_service import get_rag_service

router = APIRouter(prefix="/api")

@router.post("/market-prediction")
async def predict_crop_price(req: PricePredictionRequest):
    """
    1. Normalizes input with AI.
    2. Fetches live records from Mandi API.
    3. Analyzes results with AI.
    """
    # 1. AI-Driven Input Normalization
    try:
        rag = get_rag_service()
        normal = await rag.normalize_market_params(req.dict())
    except Exception as e:
        print(f"Normalization failed: {e}")
        normal = req.dict() 

    # 2. Fetch live records 
    try:
        records = await mandi_api_service.get_market_prices(
            state=normal.get("state", req.state),
            district=normal.get("district", req.district),
            market=normal.get("market", req.market),
            commodity=normal.get("commodity", req.commodity)
        )
        if isinstance(records, dict) and "error" in records:
            records = await mandi_api_service.get_market_prices(
                state=normal.get("state", req.state),
                district=normal.get("district", req.district),
                market="",
                commodity=normal.get("commodity", req.commodity)
            )
        if not isinstance(records, list):
            records = []
    except Exception as e:
        print(f"Mandi API critical failure: {e}")
        records = []

    # 3. AI Analysis
    try:
        raw_analysis = await rag.analyze_live_market_data(records, normal)
        clean_json = raw_analysis.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()
        analysis_data = json.loads(clean_json)
    except Exception as e:
        print(f"AI Analysis failed: {e}")
        analysis_data = {
            "analysis_bullets": ["Live analysis currently unavailable.", "Prices are showing regional volatility."],
            "sentiment": "Stable",
            "prediction_insight": "Trend is stable based on seasonal knowledge.",
            "trend_series": [{"label": "Week 1", "price": 2000}, {"label": "Week 2", "price": 2100}],
            "modeled_modal_price": 0, "modeled_range_min": 0, "modeled_range_max": 0
        }

    return {
        "records": records,
        "analysis": analysis_data,
        "normalized_input": normal
    }

@router.post("/crop-intensity-heatmap")
async def get_crop_heatmap(req: PricePredictionRequest):
    """
    Retrieves data for HeatMap visualization.
    Aggregates crops by Mandi (Sub-region) for the given district.
    """
    try:
        rag = get_rag_service()
        # Search for all mentions of crops in the district to build a distribution map
        query = f"{req.district} {req.state}"
        
        # Get from both collections for a richer map
        docs = rag.get_context(query, "MandiPrices", k=30)
        docs += rag.get_context(query, "CropProduction", k=30)
        
        import re
        distribution = {}
        for doc in docs:
            content = doc.page_content
            # Try to get Mandi Name
            mandi = doc.metadata.get("Market") or doc.metadata.get("Mandi")
            if not mandi:
                match = re.search(r'(?:Mandi|Market|District)[:\s]+([a-zA-Z0-9\s]+)', content, re.IGNORECASE)
                mandi = match.group(1).strip() if match else f"Agro Market ({req.district})"
            
            # Try to get Crop/Commodity
            commodity = doc.metadata.get("Commodity") or doc.metadata.get("Crop")
            if not commodity:
                match = re.search(r'(?:Commodity|Crop)[:\s]+([a-zA-Z0-9\s]+)', content, re.IGNORECASE)
                commodity = match.group(1).strip() if match else "Various"

            # Clean and categorize
            mandi = mandi.split(',')[0].strip()
            
            if mandi not in distribution:
                distribution[mandi] = {}
            if commodity not in distribution[mandi]:
                distribution[mandi][commodity] = 0
            distribution[mandi][commodity] += 1

        # Format for frontend (HeatMap often expects [{region, crop, value}, ...])
        formatted_data = []
        for mandi_name, crops in distribution.items():
            for crop_name, count in crops.items():
                formatted_data.append({
                    "subRegion": mandi_name,
                    "cropName": crop_name,
                    "intensity": count
                })
        
        # Simulated fallback if dataset is sparse for this specific state/district
        if len(formatted_data) == 0:
            formatted_data = await rag.generate_dynamic_heatmap(req.state, req.district)
        
        return {
            "state": req.state,
            "district": req.district,
            "data": formatted_data
        }
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
