"""Our own growing mandi price history in MongoDB (collection: mandi_prices).

The live API only keeps a short window of history. Every row we fetch is
saved here (upsert by the API's row id), so each day the project collects more
history and the forecasting model gets more data to learn from.
If MongoDB is offline these functions do nothing and return [].
"""

import re
from threading import Lock

from pymongo import ASCENDING, UpdateOne
from pymongo.errors import PyMongoError

from core import database

COLLECTION = "mandi_prices"
_indexed = False
_lock = Lock()


def _db():
    global _indexed
    db = database.get_db()
    if db is None:
        return None
    if not _indexed:
        with _lock:
            if not _indexed:
                db[COLLECTION].create_index([("id", ASCENDING)], unique=True, name="uniq_api_id")
                db[COLLECTION].create_index([("state", ASCENDING), ("commodity", ASCENDING), ("market", ASCENDING),
                                             ("arrival_date", ASCENDING)], name="by_series")
                _indexed = True
    return db


def save_rows(rows):
    """Upsert raw API rows (must have 'id'). Returns how many were written."""
    rows = [r for r in rows or [] if r.get("id") is not None and r.get("modal_price")]
    if not rows:
        return 0
    try:
        db = _db()
        if db is None:
            return 0
        ops = []
        for r in rows:
            doc = {k: r.get(k) for k in ("id", "state", "district", "market", "commodity", "variety", "grade",
                                         "arrival_date", "min_price", "max_price", "modal_price")}
            doc["market"] = str(doc["market"] or "").strip()
            ops.append(UpdateOne({"id": doc["id"]}, {"$set": doc}, upsert=True))
        result = db[COLLECTION].bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    except PyMongoError as exc:
        database.mark_unavailable(exc)
        return 0


def load_rows(state, commodity, market=None, limit=5000):
    try:
        db = _db()
        if db is None:
            return []
        query = {"state": state, "commodity": commodity}
        if market:
            query["market"] = {"$regex": f"^{re.escape(market.strip())}", "$options": "i"}
        return list(db[COLLECTION].find(query, {"_id": 0}).sort("arrival_date", ASCENDING).limit(limit))
    except PyMongoError as exc:
        database.mark_unavailable(exc)
        return []
