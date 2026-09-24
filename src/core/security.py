from datetime import datetime,timedelta,timezone
from src.core.config import settings

import jwt
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()

def passwordhasher(password:str)->str:
    return password_hash.hash(password)

def verifypasswordhas(password:str,hash_password)->bool:
    return password_hash.verify(password,hash_password)


def genrateToken(user_id:str,email:str):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload={
        "sub":user_id,
        "email":email,
        "exp":expire,
    }
    token=jwt.encode(payload,
                     settings.JWT_SECRET,
                     algorithm=settings.JWT_ALGORITHM)
    return token



def decodeToken(token:str):
    payload=jwt.decode(token,
                       settings.JWT_SECRET,
                       algorithms=[settings.JWT_ALGORITHM])
    return payload
