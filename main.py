import os
import sys
import warnings
from contextlib import asynccontextmanager
from threading import Thread

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"uvicorn\.protocols\.websockets\..*")

# 1. Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Backend imports
from api import auth, recommendation, prediction, chat, history, weather, dashboard
from core import database
from core.config import settings
from services.agri_data_service import get_agri_data_service
from services.rag_service import get_rag_service
from services.ml_service import ml_service


def warm_dataset_cache():
    try:
        get_agri_data_service().warm_up()
        print("Dataset cache warmed.")
    except Exception as exc:
        print(f"Dataset cache warm-up skipped: {exc}")
    try:
        from services.personalization_service import warm_up
        warm_up()
        print("ML models (soil Random Forest + XGBoost) loaded.")
    except Exception as exc:
        print(f"ML warm-up skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Backend started. Server is live.")
    if settings.RAG_ENABLED:
        print("RAG mode is enabled. If Weaviate is unavailable, endpoints will use dataset fallback.")
    else:
        print("Dataset mode is active. Set ENABLE_RAG=true to use Groq + Weaviate.")
    Thread(target=warm_dataset_cache, daemon=True).start()
    Thread(target=ml_service.load, daemon=True).start()
    # Connect to MongoDB Atlas in the background so startup stays fast.
    Thread(target=database.get_db, daemon=True).start()
    print("TIP: You can trigger data indexing at /api/admin/ingest")
    yield
    database.close()


app = FastAPI(title="Crop AI RAG Backend", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(recommendation.router)
app.include_router(prediction.router)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(weather.router)
app.include_router(dashboard.router)

def run_ingestion():
    """Heavy task to index data in Weaviate"""
    try:
        rag = get_rag_service()
        if not rag.available:
            print("Background Ingestion skipped: RAG service is unavailable.")
            return

        data_dir = settings.DATA_DIR
        mandi_path = os.path.join(data_dir, "mandi_data_2000_rows.csv")
        prod_path = os.path.join(data_dir, "Indian_crop_production_yield_dataset.csv")
        
        if os.path.exists(mandi_path):
            rag.ingest_csv_data(mandi_path, "MandiPrices")
        if os.path.exists(prod_path):
            rag.ingest_csv_data(prod_path, "CropProduction")
            
        print("Background Ingestion: Logic finished.")
    except Exception as e:
        print(f"Background Ingestion Error: {e}")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "services": {
            "database": database.db_status(),
            "advisory": "rag" if settings.RAG_ENABLED else "offline_dataset",
            "chat": "rag" if settings.RAG_ENABLED else "offline_dataset",
            "market_api": bool(settings.MANDI_API_KEY),
            "market_api_sample_key": settings.MANDI_API_KEY == settings.MANDI_SAMPLE_KEY,
            "weather_api": bool(settings.OPENWEATHER_API_KEY),
            "ml_models": "ready" if ml_service.ready else ("error: " + ml_service.error if ml_service.error else "loading"),
            "llm": bool(settings.LLM_ENABLED),
            "weather": bool(settings.OPENWEATHER_API_KEY),
        },
    }

@app.get("/api/admin/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    if not settings.RAG_ENABLED:
        return {"message": "RAG ingestion is disabled until Groq and Weaviate are configured."}
    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion task queued successfully."}

@app.get("/")
async def root():
    return {"message": "Crop AI RAG API is live", "version": "2.1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
