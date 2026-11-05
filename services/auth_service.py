"""User accounts and login sessions.

Primary store: MongoDB Atlas (`users`, `sessions` collections).
Fallback store: data/local_users.json, used only when Atlas is unreachable.

Passwords are hashed with PBKDF2-SHA256 (120k iterations, random salt).
Session tokens are random 256-bit strings; only their SHA-256 hash is stored,
so a leaked database cannot be used to log in.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from core import database
from core.config import settings

PROFILE_FIELDS = (
    "full_name",
    "state_name",
    "district_name",
    "taluka_name",
    "soil_type",
    "farm_size_acres",
    "main_crops",
    "preferred_language",
    "water_source",
    "irrigation_type",
    "budget",
    "soil_n",
    "soil_p",
    "soil_k",
    "soil_ph",
)

# A profile is "complete" once we know enough to personalise every prediction.
REQUIRED_PROFILE_FIELDS = ("state_name", "district_name", "soil_type", "farm_size_acres", "water_source", "main_crops")


class AuthError(ValueError):
    """Raised for user-facing auth problems (bad password, duplicate email...)."""


def _now():
    return datetime.now(timezone.utc)


def _normalize_email(email):
    return str(email or "").strip().lower()


def _hash_password(password):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"{base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def _verify_password(password, packed_hash):
    try:
        salt_part, digest_part = packed_hash.split("$", 1)
        salt = base64.b64decode(salt_part.encode())
        expected = base64.b64decode(digest_part.encode())
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(actual, expected)


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def public_user(user):
    return {
        "id": user["id"],
        "email": user["email"],
        "user_metadata": {"full_name": user.get("full_name", "")},
        "created_at": _iso(user.get("created_at")),
    }


def public_profile(user):
    profile = {"id": user["id"], "email": user["email"]}
    for field in PROFILE_FIELDS:
        profile[field] = user.get(field, [] if field == "main_crops" else None)
    profile["full_name"] = user.get("full_name") or ""
    profile["state_name"] = user.get("state_name") or ""
    profile["district_name"] = user.get("district_name") or ""
    profile["profile_complete"] = all(profile.get(field) for field in REQUIRED_PROFILE_FIELDS)
    profile["updated_at"] = _iso(user.get("updated_at"))
    return profile


def _clean_profile(data):
    clean = {}
    for field in PROFILE_FIELDS:
        if field not in data or data[field] is None:
            continue
        value = data[field]
        if field == "main_crops":
            if isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            value = [str(item).strip() for item in value if str(item).strip()][:20]
        elif field in ("farm_size_acres", "budget", "soil_n", "soil_p", "soil_k", "soil_ph"):
            value = float(value)
            if value < 0:
                raise AuthError("Farm size and budget cannot be negative.")
        else:
            value = str(value).strip()
        clean[field] = value
    return clean


# --------------------------------------------------------------------------
# Local JSON fallback (offline mode)
# --------------------------------------------------------------------------
class _LocalStore:
    def __init__(self):
        self._lock = Lock()
        self.path = settings.DATA_DIR / "local_users.json"

    def read(self):
        if not self.path.exists():
            return {"users": [], "sessions": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"users": [], "sessions": []}
        data.setdefault("users", [])
        data.setdefault("sessions", [])
        return data

    def write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, default=_iso), encoding="utf-8")

    def find_user(self, data, **match):
        key, value = next(iter(match.items()))
        return next((user for user in data["users"] if user.get(key) == value), None)


class AuthService:
    def __init__(self):
        self.local = _LocalStore()
        self._migrated = False

    # ---------------- helpers ----------------
    def _db(self):
        db = database.get_db()
        if db is not None and not self._migrated:
            self._migrate_local_users(db)
        return db

    def _migrate_local_users(self, db):
        """Copy accounts created while offline into Atlas (once per process)."""
        self._migrated = True
        try:
            data = self.local.read()
            for user in data["users"]:
                doc = {k: v for k, v in user.items()}
                doc.setdefault("created_at", _now())
                doc.setdefault("updated_at", _now())
                db[database.USERS].update_one({"email": doc["email"]}, {"$setOnInsert": doc}, upsert=True)
            if data["users"]:
                print(f"Synced {len(data['users'])} offline account(s) to MongoDB.")
        except PyMongoError as exc:
            print(f"Offline account sync skipped: {exc}")

    def _new_session(self, user):
        token = secrets.token_urlsafe(32)
        expires_at = _now() + timedelta(days=settings.SESSION_TTL_DAYS)
        session = {
            "token_hash": _hash_token(token),
            "user_id": user["id"],
            "created_at": _now(),
            "expires_at": expires_at,
        }
        return token, session

    def _session_payload(self, user, token, expires_at, provider):
        return {
            "message": "Success",
            "session": {
                "access_token": token,
                "token_type": "bearer",
                "expires_at": _iso(expires_at),
                "user": public_user(user),
            },
            "user": public_user(user),
            "profile": public_profile(user),
            "provider": provider,
        }

    # ---------------- public API ----------------
    def signup(self, name, email, password, state_name, district_name, **extra):
        email = _normalize_email(email)
        if len(password or "") < 6:
            raise AuthError("Password must be at least 6 characters.")
        if not str(name or "").strip():
            raise AuthError("Name is required.")

        now = _now()
        user = {
            "id": str(uuid4()),
            "email": email,
            "password_hash": _hash_password(password),
            "created_at": now,
            "updated_at": now,
            "last_login_at": now,
            **_clean_profile({
                "full_name": name,
                "state_name": state_name,
                "district_name": district_name,
                **extra,
            }),
        }
        token, session = self._new_session(user)

        db = self._db()
        if db is not None:
            try:
                db[database.USERS].insert_one(dict(user))
                db[database.SESSIONS].insert_one(session)
                return self._session_payload(user, token, session["expires_at"], "mongodb")
            except DuplicateKeyError as exc:
                raise AuthError("An account with this email already exists.") from exc
            except PyMongoError as exc:
                database.mark_unavailable(exc)

        with self.local._lock:
            data = self.local.read()
            if self.local.find_user(data, email=email):
                raise AuthError("An account with this email already exists.")
            data["users"].append(user)
            data["sessions"].append({k: _iso(v) for k, v in session.items()})
            self.local.write(data)
        return self._session_payload(user, token, session["expires_at"], "local")

    def login(self, email, password):
        email = _normalize_email(email)
        db = self._db()
        if db is not None:
            try:
                user = db[database.USERS].find_one({"email": email}, {"_id": 0})
                if not user or not _verify_password(password, user.get("password_hash", "")):
                    raise AuthError("Invalid email or password.")
                token, session = self._new_session(user)
                db[database.SESSIONS].insert_one(session)
                db[database.USERS].update_one({"id": user["id"]}, {"$set": {"last_login_at": _now()}})
                return self._session_payload(user, token, session["expires_at"], "mongodb")
            except PyMongoError as exc:
                database.mark_unavailable(exc)

        with self.local._lock:
            data = self.local.read()
            user = self.local.find_user(data, email=email)
            if not user or not _verify_password(password, user.get("password_hash", "")):
                raise AuthError("Invalid email or password.")
            token, session = self._new_session(user)
            now = _now()
            data["sessions"] = [s for s in data["sessions"] if _iso(s.get("expires_at")) > now.isoformat()]
            data["sessions"].append({k: _iso(v) for k, v in session.items()})
            self.local.write(data)
        return self._session_payload(user, token, session["expires_at"], "local")

    def user_from_token(self, token):
        """Return the user dict for a valid, unexpired token, else None."""
        if not token:
            return None
        token_hash = _hash_token(token)
        now = _now()

        db = self._db()
        if db is not None:
            try:
                session = db[database.SESSIONS].find_one({"token_hash": token_hash, "expires_at": {"$gt": now}})
                if session:
                    return db[database.USERS].find_one({"id": session["user_id"]}, {"_id": 0, "password_hash": 0})
            except PyMongoError as exc:
                database.mark_unavailable(exc)

        data = self.local.read()
        session = next(
            (s for s in data["sessions"] if s.get("token_hash") == token_hash and _iso(s.get("expires_at")) > now.isoformat()),
            None,
        )
        if not session:
            return None
        user = self.local.find_user(data, id=session["user_id"])
        if user:
            user = {k: v for k, v in user.items() if k != "password_hash"}
        return user

    def logout(self, token):
        token_hash = _hash_token(token or "")
        db = self._db()
        if db is not None:
            try:
                db[database.SESSIONS].delete_one({"token_hash": token_hash})
            except PyMongoError as exc:
                database.mark_unavailable(exc)
        with self.local._lock:
            data = self.local.read()
            before = len(data["sessions"])
            data["sessions"] = [s for s in data["sessions"] if s.get("token_hash") != token_hash]
            if len(data["sessions"]) != before:
                self.local.write(data)

    def update_profile(self, user_id, changes):
        clean = _clean_profile(changes)
        if not clean:
            raise AuthError("Nothing to update.")
        clean["updated_at"] = _now()

        db = self._db()
        if db is not None:
            try:
                user = db[database.USERS].find_one_and_update(
                    {"id": user_id},
                    {"$set": clean},
                    projection={"_id": 0, "password_hash": 0},
                    return_document=ReturnDocument.AFTER,
                )
                if not user:
                    raise AuthError("User not found.")
                return public_profile(user)
            except PyMongoError as exc:
                database.mark_unavailable(exc)

        with self.local._lock:
            data = self.local.read()
            user = self.local.find_user(data, id=user_id)
            if not user:
                raise AuthError("User not found.")
            user.update(clean)
            self.local.write(data)
        return public_profile(user)

    def change_password(self, user_id, current_password, new_password):
        if len(new_password or "") < 6:
            raise AuthError("New password must be at least 6 characters.")

        db = self._db()
        if db is not None:
            try:
                user = db[database.USERS].find_one({"id": user_id})
                if not user or not _verify_password(current_password, user.get("password_hash", "")):
                    raise AuthError("Current password is incorrect.")
                db[database.USERS].update_one(
                    {"id": user_id},
                    {"$set": {"password_hash": _hash_password(new_password), "updated_at": _now()}},
                )
                # Log out every other device.
                db[database.SESSIONS].delete_many({"user_id": user_id})
                return
            except PyMongoError as exc:
                database.mark_unavailable(exc)

        with self.local._lock:
            data = self.local.read()
            user = self.local.find_user(data, id=user_id)
            if not user or not _verify_password(current_password, user.get("password_hash", "")):
                raise AuthError("Current password is incorrect.")
            user["password_hash"] = _hash_password(new_password)
            data["sessions"] = [s for s in data["sessions"] if s.get("user_id") != user_id]
            self.local.write(data)


auth_service = AuthService()
