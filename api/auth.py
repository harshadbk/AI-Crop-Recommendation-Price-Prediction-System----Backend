from fastapi import APIRouter, HTTPException
from models.request_models import SignupRequest, LoginRequest
from core.config import settings
from supabase import create_client, Client

router = APIRouter(prefix="/auth")

# Initialize client using centralized settings
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@router.post("/signup")
async def signup(req: SignupRequest):
    try:
        auth_response = supabase.auth.sign_up({
            "email": req.email, "password": req.password,
            "options": {"data": {"full_name": req.name}}
        })
        if not auth_response.user: raise HTTPException(status_code=400, detail="Signup failed")
        
        user_id = auth_response.user.id
        profile_data = {"id": user_id, "full_name": req.name, "state_name": req.state_name, "district_name": req.district_name}
        try:
            supabase.table("profiles").insert(profile_data).execute()
        except: pass 
        
        return {"message": "Success", "user": auth_response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(req: LoginRequest):
    try:
        res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        profile = supabase.table("profiles").select("*").eq("id", res.user.id).single().execute()
        return {"session": res.session, "user": res.user, "profile": profile.data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
