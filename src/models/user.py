from datetime import datetime, timezone


def create_user_document(
    name: str,
    email: str,
    password_hash: str
):
    return {
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
        "is_active": True,
    }