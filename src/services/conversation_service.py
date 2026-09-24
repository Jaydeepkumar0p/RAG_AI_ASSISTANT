import uuid

from src.database.mongodb import db


# ======================================================
# CREATE CONVERSATION
# ======================================================

def create_conversation(user_id: str):

    conversation_id = str(uuid.uuid4())

    db["conversations"].insert_one({
        "conversation_id": conversation_id,
        "user_id": user_id,
        "messages": []
    })

    return conversation_id


# ======================================================
# GET CONVERSATION
# ======================================================

def get_conversation(
    conversation_id: str,
    user_id: str
):

    return db["conversations"].find_one({
        "conversation_id": conversation_id,
        "user_id": user_id
    })


# ======================================================
# GET HISTORY
# ======================================================

def get_history(
    conversation_id: str,
    user_id: str
):

    conversation = get_conversation(
        conversation_id=conversation_id,
        user_id=user_id
    )

    if not conversation:
        return []

    return conversation.get(
        "messages",
        []
    )


# ======================================================
# SAVE MESSAGE
# ======================================================

def save_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str
):

    db["conversations"].update_one(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        },
        {
            "$push": {
                "messages": {
                    "role": role,
                    "content": content
                }
            }
        }
    )