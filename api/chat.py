from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional
from core.config import settings
from services.agri_data_service import get_agri_data_service
from services.rag_service import get_rag_service
from core.security import get_optional_user
from services.history_service import StorageUnavailable, history_service
from services.personalization_service import farmer_context
from services import advisor_service
from services.llm_service import llm_service

router = APIRouter(prefix="/api")


def get_ready_rag_service():
    if not settings.RAG_ENABLED:
        return None

    rag = get_rag_service()
    return rag if rag.available else None

class ChatMessage(BaseModel):
    id: Optional[str] = None
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

def _save_chat(user, question, response):
    """Store the question and answer for logged-in users; never break the chat if this fails."""
    if not user or not isinstance(response, dict):
        return
    now = datetime.now(timezone.utc)
    try:
        history_service.add_chat_messages(user["id"], [
            {"role": "user", "text": question, "ts": now.isoformat()},
            # +1ms so the answer always sorts after the question.
            {"role": "assistant", "text": response.get("reply", ""), "ts": (now + timedelta(milliseconds=1)).isoformat()},
        ])
    except StorageUnavailable as exc:
        print(f"Chat not saved: {exc}")


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, user=Depends(get_optional_user)):
    response = await _chat_reply(req, user)
    await run_in_threadpool(_save_chat, user, req.message, response)
    return response


async def _chat_reply(req: ChatRequest, user=None):
    """Farmer advisory chat: Groq LLM grounded in live weather + ML predictions,
    then RAG, then the offline dataset reply."""
    offline_service = get_agri_data_service()
    context = farmer_context(user)  # farmer's district, soil and crops for the fallbacks
    history_list = [{"role": m.role, "content": m.text} for m in req.history or []]
    if llm_service.enabled:
        try:
            return await advisor_service.chat(req.message, history_list, user)
        except Exception as e:
            print(f"Live LLM chat unavailable, falling back: {e}")
    rag = get_ready_rag_service()
    if rag is None:
        return offline_service.chat_reply(req.message, [{"role": m.role, "content": m.text} for m in req.history or []], context)

    try:
        # Convert history objects to simple list for LLM context
        history_list = []
        if req.history:
            history_list = [{"role": m.role, "content": m.text} for m in req.history]
            
        response = await rag.chat_with_context(req.message, history_list, context)
        return response
    except Exception as e:
        print(f"AI chat unavailable, using dataset reply: {e}")
        return offline_service.chat_reply(req.message, [{"role": m.role, "content": m.text} for m in req.history or []], context)
