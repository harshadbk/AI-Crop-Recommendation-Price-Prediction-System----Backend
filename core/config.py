import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    def __init__(self):
        self.BACKEND_DIR = BACKEND_DIR
        self.DATA_DIR = BACKEND_DIR / "data"
        # MongoDB Atlas
        self.MONGODB_URI = self._get("MONGODB_URI")
        self.MONGODB_USERNAME = self._get("MONGODB_USERNAME")
        self.MONGODB_PASSWORD = self._get("MONGODB_PASSWORD")
        self.MONGODB_DB_NAME = self._get("MONGODB_DB_NAME", "crop_ai")
        self.MONGODB_TIMEOUT_MS = int(self._get_float("MONGODB_TIMEOUT_MS", 8000))
        self.SESSION_TTL_DAYS = int(self._get_float("SESSION_TTL_DAYS", 7))
        self.HISTORY_LIMIT = int(self._get_float("HISTORY_LIMIT", 200))
        self.ENABLE_RAG = self._get_bool("ENABLE_RAG", False)
        self.GROQ_API_KEY = self._get("GROQ_API_KEY")
        self.GROQ_MODEL = self._get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.GROQ_FALLBACK_MODEL = self._get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
        self.LLM_TIMEOUT_SECONDS = self._get_float("LLM_TIMEOUT_SECONDS", 25)
        # OpenWeatherMap
        self.OPENWEATHER_API_KEY = self._get("OPENWEATHER_API_KEY")
        self.WEATHER_CACHE_SECONDS = self._get_float("WEATHER_CACHE_SECONDS", 600)
        self.ML_DIR = BACKEND_DIR / "ml" / "artifacts"
        self.WEAVIATE_URL = self._get("WEAVIATE_CLUSTER")
        self.WEAVIATE_API_KEY = self._get("WEAVIATE_API_KEY")
        # Public sample key from data.gov.in (max 10 rows). Put your own key in .env.
        self.MANDI_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
        self.MANDI_API_KEY = self._get("MANDI_API_KEY", self.MANDI_SAMPLE_KEY)
        self.MANDI_RESOURCE_ID = self._get(
            "MANDI_RESOURCE_ID",
            "9ef84268-d588-465a-a308-a864a43d0070",
        )
        self.FAST_MARKET_LOCAL_FIRST = self._get_bool("FAST_MARKET_LOCAL_FIRST", True)
        self.MANDI_API_TIMEOUT_SECONDS = self._get_float("MANDI_API_TIMEOUT_SECONDS", 0.75)
        self.MANDI_API_CACHE_SECONDS = self._get_float("MANDI_API_CACHE_SECONDS", 600.0)
        # Real-time mandi prices API (free, no key). Render free tier can take ~20s to wake up.
        self.MANDI_LIVE_API_URL = self._get("MANDI_LIVE_API_URL", "https://mandi-api.onrender.com/v1/prices")
        self.MANDI_LIVE_TIMEOUT_SECONDS = self._get_float("MANDI_LIVE_TIMEOUT_SECONDS", 25.0)
        self.DASHBOARD_CACHE_SECONDS = self._get_float("DASHBOARD_CACHE_SECONDS", 600)

    @staticmethod
    def _get(name, default=None):
        value = os.getenv(name)
        if value is None:
            return default

        cleaned = value.strip()
        return cleaned if cleaned else default

    @staticmethod
    def _get_bool(name, default=False):
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _get_float(name, default):
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @property
    def MONGODB_ENABLED(self):
        return bool(self.MONGODB_URI)

    @property
    def RAG_ENABLED(self):
        return bool(self.ENABLE_RAG and self.GROQ_API_KEY and self.WEAVIATE_URL and self.WEAVIATE_API_KEY)

    @property
    def LLM_ENABLED(self):
        return bool(self.GROQ_API_KEY)


settings = Settings()
