from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.rag_service import get_rag_service

router = APIRouter(prefix="/api")

class ChatMessage(BaseModel):
    id: Optional[str] = None
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Conversational RAG endpoint for farmer advisory.
    """
    try:
        rag = get_rag_service()
        
        # Convert history objects to simple list for LLM context
        history_list = []
        if req.history:
            history_list = [{"role": m.role, "content": m.text} for m in req.history]
            
        response = await rag.chat_with_context(req.message, history_list)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
