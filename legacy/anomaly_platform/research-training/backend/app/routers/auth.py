from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models import User
from schemas import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse
from auth import authenticate_user, create_access_token, get_current_user, get_password_hash

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=access_token)


@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if username already exists
    result = await db.execute(select(User).where(User.username == request.username))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Username '{request.username}' is already taken",
        )

    # Create user
    user = User(
        username=request.username,
        password_hash=get_password_hash(request.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return RegisterResponse(
        id=user.id,
        username=user.username,
        created_at=user.created_at,
    )


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "created_at": str(current_user.created_at),
    }
