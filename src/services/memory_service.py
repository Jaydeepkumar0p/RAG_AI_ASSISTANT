from src.database.mongodb import db


def get_conversation_history(
    conversation_id: str,
    user_id: str,
    limit: int = 50
):
    messages = db["chat_messages"].find(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    ).sort(
        "created_at",
        1
    ).limit(limit)

    history = []

    for message in messages:
        history.append({
            "role": message["role"],
            "content": message["content"],
            "created_at": message["created_at"]
        })

    return history


def save_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str
):
    from datetime import datetime

    db["chat_messages"].insert_one({
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": datetime.utcnow()
    })



def get_user_conversations(
    user_id: str
):
    pipeline = [

        {
            "$match": {
                "user_id": user_id
            }
        },

        {
            "$sort": {
                "created_at": -1
            }
        },

        {
            "$group": {
                "_id": "$conversation_id",

                "last_message": {
                    "$first": "$content"
                },

                "last_role": {
                    "$first": "$role"
                },

                "last_message_at": {
                    "$first": "$created_at"
                },

                "message_count": {
                    "$sum": 1
                }
            }
        },

        {
            "$sort": {
                "last_message_at": -1
            }
        }
    ]

    conversations = db[
        "chat_messages"
    ].aggregate(pipeline)

    result = []

    for conversation in conversations:

        result.append({
            "conversation_id":
                conversation["_id"],

            "last_message":
                conversation["last_message"],

            "last_role":
                conversation["last_role"],

            "last_message_at":
                conversation["last_message_at"],

            "message_count":
                conversation["message_count"]
        })

    return result


def delete_conversation(
    conversation_id: str,
    user_id: str
):
    result = db["chat_messages"].delete_many(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )

    return result.deleted_count