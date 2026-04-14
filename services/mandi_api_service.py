import httpx
from backend.core.config import settings

class MandiApiService:
    def __init__(self):
        self.api_key = settings.MANDI_API_KEY
        self.resource_id = settings.MANDI_RESOURCE_ID
        self.base_url = "https://api.data.gov.in/resource/"

    async def get_market_prices(self, state: str, district: str, market: str, commodity: str):
        if not self.api_key:
            return {
                "error": "MANDI_API_KEY is missing in backend .env",
                "status": "failed"
            }

        url = f"{self.base_url}{self.resource_id}"
        params = {
            "api-key": self.api_key,
            "format": "json",
            "filters[state]": state,
            "filters[district]": district,
            "filters[market]": market,
            "filters[commodity]": commodity
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                records = data.get("records", [])
                
                if not records:
                    return {"message": "No data found for the given parameters", "records": []}
                
                return records
            except Exception as e:
                return {
                    "error": f"API Request Failed: {str(e)}",
                    "status": "failed"
                }

# Singleton instance
mandi_api_service = MandiApiService()
