"""Per-user history (crop recommendations, market predictions) and chat log.

Stored in MongoDB Atlas. These features need a logged-in user and a reachable
database; when Atlas is down the routes answer 503 and the frontend keeps its
browser copy, so nothing is lost.
"""

from datetime import datetime, timezone
from uuid import uuid4

from pymongo import DESCENDING, ReplaceOne
from pymongo.errors import PyMongoError

from core import database
from core.config import settings

HISTORY_KINDS = {"crop", "market", "recommendation", "prediction"}
CHAT_LIMIT = 500


class StorageUnavailable(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _db():
    db = database.get_db()
    if db is None:
        raise StorageUnavailable("Database is unavailable. Your data is kept on this device for now.")
    return db


def _parse_time(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return _now()


def _history_out(doc):
    entry = dict(doc.get("entry") or {})
    entry["id"] = doc["id"]
    entry.setdefault("kind", doc.get("kind"))
    entry["createdAt"] = entry.get("createdAt") or doc["created_at"].isoformat()
    return entry


def _guard(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PyMongoError as exc:
            database.mark_unavailable(exc)
            raise StorageUnavailable("Database request failed. Please try again.") from exc
    wrapper.__name__ = fn.__name__
    return wrapper


class HistoryService:
    # ---------------- history ----------------
    def _history_doc(self, user_id, entry):
        entry = dict(entry or {})
        entry_id = str(entry.pop("id", "") or f"h_{uuid4().hex}")
        kind = str(entry.get("kind") or entry.get("type") or "").lower() or "other"
        created = _parse_time(entry.get("generatedAt") or entry.get("createdAt") or entry.get("ts"))
        return {
            "id": entry_id,
            "user_id": user_id,
            "kind": kind,
            "crop": (
                (entry.get("input") or {}).get("commodity")
                or (entry.get("input") or {}).get("desiredCrop")
                or ""
            ),
            "entry": entry,
            "created_at": created,
            "saved_at": _now(),
        }

    @_guard
    def add(self, user_id, entry):
        doc = self._history_doc(user_id, entry)
        db = _db()
        # Upsert on (user, id) so a retried save never duplicates an entry.
        db[database.HISTORY].replace_one({"user_id": user_id, "id": doc["id"]}, doc, upsert=True)
        self._trim(db, user_id)
        return _history_out(doc)

    @_guard
    def bulk_add(self, user_id, entries):
        """Used to upload history that was stored in the browser before login."""
        docs = [self._history_doc(user_id, e) for e in (entries or [])][: settings.HISTORY_LIMIT]
        if not docs:
            return 0
        db = _db()
        ops = [ReplaceOne({"user_id": user_id, "id": d["id"]}, d, upsert=True) for d in docs]
        db[database.HISTORY].bulk_write(ops, ordered=False)
        self._trim(db, user_id)
        return len(docs)

    def _trim(self, db, user_id):
        keep = settings.HISTORY_LIMIT
        old = list(
            db[database.HISTORY]
            .find({"user_id": user_id}, {"_id": 1})
            .sort("created_at", DESCENDING)
            .skip(keep)
        )
        if old:
            db[database.HISTORY].delete_many({"_id": {"$in": [d["_id"] for d in old]}})

    @_guard
    def list(self, user_id, kind=None, crop=None, date_from=None, date_to=None, limit=100, skip=0):
        query = {"user_id": user_id}
        if kind:
            query["kind"] = kind.lower()
        if crop:
            query["crop"] = crop
        if date_from or date_to:
            query["created_at"] = {}
            if date_from:
                query["created_at"]["$gte"] = _parse_time(date_from)
            if date_to:
                query["created_at"]["$lte"] = _parse_time(date_to)

        db = _db()
        total = db[database.HISTORY].count_documents(query)
        cursor = (
            db[database.HISTORY]
            .find(query, {"_id": 0})
            .sort("created_at", DESCENDING)
            .skip(max(0, int(skip)))
            .limit(max(1, min(int(limit), settings.HISTORY_LIMIT)))
        )
        return {"items": [_history_out(d) for d in cursor], "total": total}

    @_guard
    def delete(self, user_id, entry_id):
        result = _db()[database.HISTORY].delete_one({"user_id": user_id, "id": entry_id})
        return result.deleted_count

    @_guard
    def clear(self, user_id):
        return _db()[database.HISTORY].delete_many({"user_id": user_id}).deleted_count

    # ---------------- chat ----------------
    @_guard
    def add_chat_messages(self, user_id, messages):
        docs = []
        for message in messages:
            docs.append({
                "id": message.get("id") or f"m_{uuid4().hex}",
                "user_id": user_id,
                "role": message["role"],
                "text": str(message.get("text", ""))[:8000],
                "created_at": _parse_time(message.get("ts")),
            })
        if not docs:
            return []
        db = _db()
        db[database.CHAT_MESSAGES].insert_many([dict(d) for d in docs])
        # Keep only the most recent CHAT_LIMIT messages per user.
        old = list(
            db[database.CHAT_MESSAGES]
            .find({"user_id": user_id}, {"_id": 1})
            .sort("created_at", DESCENDING)
            .skip(CHAT_LIMIT)
        )
        if old:
            db[database.CHAT_MESSAGES].delete_many({"_id": {"$in": [d["_id"] for d in old]}})
        return docs

    @_guard
    def list_chat(self, user_id, limit=200):
        cursor = (
            _db()[database.CHAT_MESSAGES]
            .find({"user_id": user_id}, {"_id": 0, "user_id": 0})
            .sort("created_at", DESCENDING)
            .limit(max(1, min(int(limit), CHAT_LIMIT)))
        )
        items = [
            {"id": d["id"], "role": d["role"], "text": d["text"], "ts": d["created_at"].isoformat()}
            for d in cursor
        ]
        return list(reversed(items))

    @_guard
    def clear_chat(self, user_id):
        return _db()[database.CHAT_MESSAGES].delete_many({"user_id": user_id}).deleted_count


history_service = HistoryService()
