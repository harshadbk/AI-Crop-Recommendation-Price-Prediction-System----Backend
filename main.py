import os
import sys
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. Load env variables
load_dotenv()

# 2. Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 3. Backend imports
from api import auth, recommendation, prediction, chat
from services.rag_service import get_rag_service

app = FastAPI(title="Crop AI RAG Backend")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(recommendation.router)
app.include_router(prediction.router)
app.include_router(chat.router)

def run_ingestion():
    """Heavy task to index data in Weaviate"""
    try:
        rag = get_rag_service()
        data_dir = "data"
        mandi_path = os.path.join(data_dir, "mandi_data_2000_rows.csv")
        prod_path = os.path.join(data_dir, "Indian_crop_production_yield_dataset.csv")
        
        if os.path.exists(mandi_path):
            rag.ingest_csv_data(mandi_path, "MandiPrices")
        if os.path.exists(prod_path):
            rag.ingest_csv_data(prod_path, "CropProduction")
            
        print("Background Ingestion: Logic finished.")
    except Exception as e:
        print(f"Background Ingestion Error: {e}")

@app.on_event("startup")
async def startup_event():
    print("Backend started. Server is live.")
    print("TIP: You can trigger data indexing at /api/admin/ingest")

@app.get("/api/admin/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion task queued successfully."}

@app.get("/")
async def root():
    return {"message": "Crop AI RAG API is live", "version": "2.1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
