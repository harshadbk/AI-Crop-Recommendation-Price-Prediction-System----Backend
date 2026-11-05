from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from core.security import get_current_user
from models.request_models import HistoryBulkRequest, HistoryEntry
from services.history_service import StorageUnavailable, history_service

router = APIRouter(prefix="/api", tags=["history"])


def _dump(model):
    return model.model_dump(exclude_none=True) if hasattr(model, "model_dump") else model.dict(exclude_none=True)


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StorageUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------- history ----------------
@router.get("/history")
def list_history(
    kind: Optional[str] = None,
    crop: Optional[str] = None,
    date_from: Optional[str] = Query(default=None, alias="from"),
    date_to: Optional[str] = Query(default=None, alias="to"),
    limit: int = Query(default=100, ge=1, le=200),
    skip: int = Query(default=0, ge=0),
    user=Depends(get_current_user),
):
    return _run(history_service.list, user["id"], kind, crop, date_from, date_to, limit, skip)


@router.post("/history", status_code=201)
def add_history(entry: HistoryEntry, user=Depends(get_current_user)):
    return {"item": _run(history_service.add, user["id"], _dump(entry))}


@router.post("/history/sync")
def sync_history(req: HistoryBulkRequest, user=Depends(get_current_user)):
    """Upload entries saved in the browser (e.g. before the user logged in)."""
    count = _run(history_service.bulk_add, user["id"], [_dump(item) for item in req.items])
    return {"synced": count}


@router.delete("/history/{entry_id}")
def delete_history(entry_id: str, user=Depends(get_current_user)):
    if not _run(history_service.delete, user["id"], entry_id):
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"deleted": 1}


@router.delete("/history")
def clear_history(user=Depends(get_current_user)):
    return {"deleted": _run(history_service.clear, user["id"])}


# ---------------- chat log ----------------
@router.get("/chat/history")
def chat_history(limit: int = Query(default=200, ge=1, le=500), user=Depends(get_current_user)):
    return {"items": _run(history_service.list_chat, user["id"], limit)}


@router.delete("/chat/history")
def clear_chat_history(user=Depends(get_current_user)):
    return {"deleted": _run(history_service.clear_chat, user["id"])}
