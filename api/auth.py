from fastapi import APIRouter, Depends, HTTPException

from core.security import get_current_user, get_token
from models.request_models import (
    ChangePasswordRequest,
    LoginRequest,
    ProfileUpdateRequest,
    SignupRequest,
)
from services.auth_service import AuthError, auth_service, public_profile, public_user

router = APIRouter(prefix="/auth", tags=["auth"])


def _dump(model, exclude_unset=False):
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=exclude_unset)
    return model.dict(exclude_unset=exclude_unset)


@router.post("/signup")
def signup(req: SignupRequest):
    """Create an account and return a session, so the user is logged in immediately."""
    data = _dump(req)
    try:
        return auth_service.signup(
            name=data.pop("name"),
            email=data.pop("email"),
            password=data.pop("password"),
            state_name=data.pop("state_name"),
            district_name=data.pop("district_name"),
            **data,
        )
    except AuthError as exc:
        status = 409 if "already exists" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/login")
def login(req: LoginRequest):
    try:
        return auth_service.login(email=req.email, password=req.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me")
def me(user=Depends(get_current_user)):
    """Validate the stored token and return the current user + profile."""
    return {"user": public_user(user), "profile": public_profile(user)}


@router.post("/logout")
def logout(token=Depends(get_token)):
    if token:
        auth_service.logout(token)
    return {"message": "Logged out"}


@router.put("/profile")
def update_profile(req: ProfileUpdateRequest, user=Depends(get_current_user)):
    try:
        profile = auth_service.update_profile(user["id"], _dump(req, exclude_unset=True))
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": profile}


@router.post("/change-password")
def change_password(req: ChangePasswordRequest, user=Depends(get_current_user)):
    try:
        auth_service.change_password(user["id"], req.current_password, req.new_password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Password changed. Please log in again."}
