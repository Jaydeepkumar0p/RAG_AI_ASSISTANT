from fastapi import APIRouter,HTTPException,status
from src.database.mongodb import users_collection

from src.core.security import genrateToken,passwordhasher,decodeToken,verifypasswordhas
from src.schema.auth import UserResponse,RegisterRequest,LoginRequest,TokenResponse
from src.models.user import create_user_document

router=APIRouter(
     prefix="/auth",
    tags=["Authentication"]
)

@router.post("/register",status_code=status.HTTP_201_CREATED)
async def RegisterUser(user:RegisterRequest):
    existing_user=users_collection.find_one({"email":user.email})
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="user is already exist"
        )
    hashpassword=passwordhasher(user.password)
    new_user=create_user_document(
        email=user.email,
        name=user.name,
        password_hash=hashpassword
        
    )
    result=users_collection.insert_one(new_user)
    return{
         "message": "User registered successfully",
        "user_id": str(result.inserted_id)
    }


@router.post("/login", response_model=TokenResponse)
async def login(user: LoginRequest):

    email = user.email.lower().strip()

    db_user = users_collection.find_one({
        "email": email
    })

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verifypasswordhas(
        user.password,
        db_user["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not db_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    token = genrateToken(
        user_id=str(db_user["_id"]),
        email=db_user["email"]
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }
    

