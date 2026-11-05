"""FastAPI dependencies for reading the logged-in user from `Authorization: Bearer <token>`."""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from services.auth_service import auth_service


def get_token(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def get_optional_user(token: Optional[str] = Depends(get_token)):
    """The user if a valid token was sent, otherwise None (guest access)."""
    return auth_service.user_from_token(token) if token else None


def get_current_user(user=Depends(get_optional_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
