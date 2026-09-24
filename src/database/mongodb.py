from pymongo import MongoClient

from pymongo.server_api import ServerApi

from src.core.config import settings


# ======================================================
# CONFIG
# ======================================================

MONGO_URI = settings.MONGO_URI

DATABASE_NAME = settings.DATABASE_NAME


if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not configured"
    )


# ======================================================
# CLIENT
# ======================================================

client = MongoClient(

    MONGO_URI,

    server_api=ServerApi(
        version="1"
    ),

    serverSelectionTimeoutMS=5000,

    connectTimeoutMS=5000
)


# ======================================================
# DATABASE
# ======================================================

db = client[
    DATABASE_NAME
]


# ======================================================
# COLLECTIONS
# ======================================================

users_collection = db[
    "users"
]

documents_collection = db[
    "documents"
]

conversations_collection = db[
    "conversations"
]


# ======================================================
# HEALTH CHECK
# ======================================================

def check_database_connection():

    try:

        client.admin.command(
            "ping"
        )

        print(
            "MongoDB connected successfully"
        )

        return True

    except Exception as e:

        print(
            "MongoDB connection error:",
            e
        )

        return False