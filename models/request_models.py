from pydantic import BaseModel, EmailStr, Field
from typing import Any, Dict, List, Optional, Union


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    state_name: str = Field(..., min_length=2, max_length=80)
    district_name: str = Field(..., min_length=2, max_length=80)
    taluka_name: Optional[str] = None
    soil_type: Optional[str] = None
    farm_size_acres: Optional[float] = Field(default=None, ge=0)
    main_crops: Optional[Union[List[str], str]] = None
    preferred_language: Optional[str] = None
    water_source: Optional[str] = None
    irrigation_type: Optional[str] = None
    budget: Optional[float] = Field(default=None, ge=0)
    # Soil Health Card values (optional): N, P, K in kg/ha and soil pH
    soil_n: Optional[float] = Field(default=None, ge=0, le=500)
    soil_p: Optional[float] = Field(default=None, ge=0, le=500)
    soil_k: Optional[float] = Field(default=None, ge=0, le=1000)
    soil_ph: Optional[float] = Field(default=None, ge=3, le=10)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=120)
    state_name: Optional[str] = None
    district_name: Optional[str] = None
    taluka_name: Optional[str] = None
    soil_type: Optional[str] = None
    farm_size_acres: Optional[float] = Field(default=None, ge=0)
    main_crops: Optional[Union[List[str], str]] = None
    preferred_language: Optional[str] = None
    water_source: Optional[str] = None
    irrigation_type: Optional[str] = None
    budget: Optional[float] = Field(default=None, ge=0)
    # Soil Health Card values (optional): N, P, K in kg/ha and soil pH
    soil_n: Optional[float] = Field(default=None, ge=0, le=500)
    soil_p: Optional[float] = Field(default=None, ge=0, le=500)
    soil_k: Optional[float] = Field(default=None, ge=0, le=1000)
    soil_ph: Optional[float] = Field(default=None, ge=3, le=10)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=128)


class HistoryEntry(BaseModel):
    """One saved result. The frontend's packed object is stored as-is."""
    model_config = {"extra": "allow"}

    id: Optional[str] = None
    kind: Optional[str] = None
    createdAt: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None


class HistoryBulkRequest(BaseModel):
    items: List[HistoryEntry] = Field(default_factory=list, max_length=500)

class RecommendationRequest(BaseModel):
    type: str = "crop_suggestion"
    # Optional: when missing, the logged-in farmer's saved profile is used.
    stateName: Optional[str] = None
    districtName: Optional[str] = None
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
    # Soil test values typed on the form (override the profile for this request)
    soilN: Optional[float] = None
    soilP: Optional[float] = None
    soilK: Optional[float] = None
    soilPh: Optional[float] = None

class PricePredictionRequest(BaseModel):
    state: Optional[str] = ""
    district: Optional[str] = ""
    market: Optional[str] = ""
    commodity: Optional[str] = ""
