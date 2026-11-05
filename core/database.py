"""MongoDB Atlas connection.

One shared MongoClient for the whole app (pymongo is thread-safe and pools
connections). The client is created lazily, pinged once, and indexes are
created on first use. If Atlas cannot be reached, `get_db()` returns None and
callers fall back to the local JSON store, so the app keeps working offline.
"""

import time
from threading import Lock

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError
from pymongo.server_api import ServerApi

from core.config import settings

USERS = "users"
SESSIONS = "sessions"
HISTORY = "history"
CHAT_MESSAGES = "chat_messages"

RETRY_AFTER_SECONDS = 60

_client = None
_db = None
_last_failure = 0.0
_lock = Lock()


def _ensure_indexes(db):
    db[USERS].create_index([("email", ASCENDING)], unique=True, name="uniq_email")

    db[SESSIONS].create_index([("token_hash", ASCENDING)], unique=True, name="uniq_token")
    db[SESSIONS].create_index([("user_id", ASCENDING)], name="by_user")
    # TTL index: MongoDB deletes a session automatically once expires_at passes.
    db[SESSIONS].create_index([("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl_expiry")

    db[HISTORY].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)], name="by_user_recent")
    db[HISTORY].create_index([("user_id", ASCENDING), ("client_id", ASCENDING)], name="by_user_client_id")

    db[CHAT_MESSAGES].create_index([("user_id", ASCENDING), ("created_at", ASCENDING)], name="by_user_time")


def get_db():
    """Return the Atlas database, or None if MongoDB is not configured/reachable."""
    global _client, _db, _last_failure

    if _db is not None:
        return _db
    if not settings.MONGODB_ENABLED:
        return None
    # After a failure, don't hammer Atlas on every request.
    if _last_failure and time.time() - _last_failure < RETRY_AFTER_SECONDS:
        return None

    with _lock:
        if _db is not None:
            return _db
        client = None
        try:
            client = MongoClient(
                settings.MONGODB_URI,
                server_api=ServerApi("1"),  # same as mongosh --apiVersion 1
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                retryWrites=True,
                tz_aware=True,
                appname="crop-ai-backend",
            )
            client.admin.command("ping")
            db = client[settings.MONGODB_DB_NAME]
            _ensure_indexes(db)
            _client, _db = client, db
            _last_failure = 0.0
            print(f"MongoDB connected: database '{settings.MONGODB_DB_NAME}'.")
            return _db
        except PyMongoError as exc:
            if client is not None:
                client.close()
            _last_failure = time.time()
            print(f"MongoDB unavailable, using local fallback store: {exc}")
            return None


def mark_unavailable(exc=None):
    """Called when an operation fails mid-flight so the next call reconnects."""
    global _client, _db, _last_failure
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
        _client, _db = None, None
        _last_failure = time.time()
    if exc is not None:
        print(f"MongoDB operation failed, switching to local fallback: {exc}")


def db_status():
    db = get_db()
    return "mongodb" if db is not None else "local_file"


def close():
    global _client, _db
    with _lock:
        if _client is not None:
            _client.close()
        _client, _db = None, None
