from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from src.middleware.authmiddleware import get_current_active_user
from src.database.mongodb import db


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


# ======================================================
# GET ALL CONVERSATIONS
# ======================================================

@router.get("")
async def get_conversations(
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    conversations = db["conversations"].find(
        {
            "user_id": user_id
        }
    )

    result = []

    for conversation in conversations:

        messages = conversation.get(
            "messages",
            []
        )

        result.append({
            "conversation_id": conversation[
                "conversation_id"
            ],
            "message_count": len(messages),
            "created_at": conversation.get(
                "created_at"
            ),
            "updated_at": conversation.get(
                "updated_at"
            )
        })

    return {
        "conversations": result
    }


# ======================================================
# GET SINGLE CONVERSATION
# ======================================================

@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    conversation = db["conversations"].find_one(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )

    if not conversation:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    return {
        "conversation_id": conversation[
            "conversation_id"
        ],
        "messages": conversation.get(
            "messages",
            []
        ),
        "created_at": conversation.get(
            "created_at"
        ),
        "updated_at": conversation.get(
            "updated_at"
        )
    }


# ======================================================
# DELETE CONVERSATION
# ======================================================

@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    result = db["conversations"].delete_one(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )

    if result.deleted_count == 0:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id
    }