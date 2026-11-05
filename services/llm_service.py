"""Groq LLM used directly (no vector DB needed) for chat and for explaining
ML recommendations. The LLM is handed live weather, ML predictions and dataset
prices as context and told to use only those numbers."""

import asyncio
import json
import re

from core.config import settings

SYSTEM_CHAT = (
    "You are 'Crop AI Expert', an agricultural advisor for Indian farmers. "
    "Answer only farming topics (crops, soil, weather, irrigation, fertilizer, pests, diseases, markets, "
    "farm finance and government schemes); politely decline anything else. "
    "You receive LIVE CONTEXT built from real-time OpenWeather data, trained XGBoost crop/yield/price models "
    "and mandi datasets. Prefer those facts and quote their numbers (temperatures, rain, prices, scores); "
    "never invent prices or forecasts that are not in the context, and say when data is missing. "
    "Use general agronomy knowledge for practices (doses, spacing, schedules). "
    "Reply in the language the farmer writes in. Answer the specific question first, then give a few concise, "
    "practical steps the farmer can act on. Be clear about uncertainty and ask for relevant missing details "
    "when they would change the advice. Keep the reply easy to scan using short paragraphs and plain-text "
    "bullets starting with '- '. Do not use markdown bold or emphasis markers such as ** or __, markdown "
    "tables, or code fences. "
    'Return ONLY a JSON object: {"reply": "<plain-text answer>", "suggested": ["<3 short, relevant follow-up questions>"]}'
)

SYSTEM_EXPLAIN = (
    "You explain crop recommendations to an Indian farmer. You get the farm inputs, live weather and the "
    "ML ranking with scores, predicted yields, prices and cautions. Write 4-6 short bullet sentences: why the "
    "top crop wins, how the current weather affects sowing, the main risk, and one action for this week. "
    "Use only numbers given in the data. "
    'Return ONLY JSON: {"bullets": ["..."]}'
)


def _parse_json(text):
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class LLMService:
    def __init__(self):
        self._client = None

    @property
    def enabled(self):
        return settings.LLM_ENABLED

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS)
        return self._client

    async def _complete(self, messages, max_tokens=1500):
        client = self._get_client()
        last_error = None
        for model in dict.fromkeys([settings.GROQ_MODEL, settings.GROQ_FALLBACK_MODEL]):
            if not model:
                continue
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model, messages=messages, temperature=0.3, max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    ),
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                return resp.choices[0].message.content
            except Exception as exc:  # try the fallback model
                last_error = exc
                print(f"LLM model {model} failed: {exc}")
        raise RuntimeError(f"LLM unavailable: {last_error}")

    async def chat(self, message, history, context):
        messages = [{"role": "system", "content": SYSTEM_CHAT}]
        for item in (history or [])[-8:]:
            role = "assistant" if item.get("role") in {"assistant", "bot", "ai"} else "user"
            content = item.get("content") or item.get("text") or ""
            if content:
                messages.append({"role": role, "content": content[:2000]})
        messages.append({"role": "user", "content": f"LIVE CONTEXT:\n{context}\n\nFARMER QUESTION: {message}"})
        raw = await self._complete(messages)
        data = _parse_json(raw)
        if not data or not data.get("reply"):
            return {"reply": (raw or "").strip(), "suggested": []}
        suggested = [str(s) for s in (data.get("suggested") or [])][:3]
        return {"reply": str(data["reply"]), "suggested": suggested}

    async def explain(self, facts):
        raw = await self._complete([
            {"role": "system", "content": SYSTEM_EXPLAIN},
            {"role": "user", "content": json.dumps(facts, default=str)[:12000]},
        ], max_tokens=900)
        data = _parse_json(raw) or {}
        bullets = [str(b) for b in data.get("bullets") or [] if str(b).strip()]
        if not bullets:
            raise RuntimeError("LLM returned no bullets")
        return bullets[:6]


llm_service = LLMService()
