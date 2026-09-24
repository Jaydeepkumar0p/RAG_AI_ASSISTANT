from pydantic import BaseModel,EmailStr

class RegisterRequest(BaseModel):
    name:str
    email:EmailStr
    password: str


class LoginRequest(BaseModel):
     email: EmailStr
     password:str

class UserResponse(BaseModel):
     id:str
     email:EmailStr
     name:str


class TokenResponse(BaseModel):

    access_token: str
    token_type: str