from fastapi import APIRouter, Depends

from src.middleware.authmiddleware import get_current_active_user


router = APIRouter(
    prefix="/users",
    tags=["Users"]
)


@router.get("/me")
async def get_current_user(
    current_user=Depends(get_current_active_user)
):

    return {
        "id": str(current_user["_id"]),
        "name": current_user["name"],
        "email": current_user["email"],
        "is_active": current_user["is_active"],
    }