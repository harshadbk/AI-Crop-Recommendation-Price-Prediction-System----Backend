from pydantic import BaseModel, EmailStr
from typing import Optional

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    state_name: str
    district_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RecommendationRequest(BaseModel):
    type: str = "crop_suggestion"
    stateName: str
    districtName: str
    season: Optional[str] = None
    soilType: Optional[str] = None
    waterSource: Optional[str] = None
    irrigationType: Optional[str] = None
    budget: Optional[float] = None
    landSize: Optional[float] = None
    targetedCrop: Optional[str] = None
    marketName: Optional[str] = None
    commodity: Optional[str] = None
    variety: Optional[str] = None
    priceDate: Optional[str] = None

class PricePredictionRequest(BaseModel):
    state: str
    district: str
    market: str
    commodity: str
