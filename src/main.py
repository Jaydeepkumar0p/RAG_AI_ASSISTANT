from fastapi import FastAPI

from src.database.mongodb import check_database_connection
from src.routes.auth import router as auth_router
from src.routes.user import router as user_router
from src.routes.document import router as document_router
from src.routes.chat import router as chat_router
from src.services.rag_service import create_collection

from src.routes.conversation import router as conversation_router
from src.database.qdrant import (
    client,
    COLLECTION_NAME
)





app = FastAPI(
    title="AI Study Assistant API",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://YOUR-FRONTEND-DOMAIN.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    create_collection()


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(document_router)
app.include_router(chat_router)
app.include_router(conversation_router)

@app.get("/")
def root():
    return {
        "message": "AI Study Assistant API is running"
    }


@app.get("/health/db")
def database_health():

    connected = check_database_connection()

    if connected:
        return {
            "database": "MongoDB",
            "status": "connected"
        }

    return {
        "database": "MongoDB",
        "status": "disconnected"
    }


@app.get("/health/qdrant")
def qdrant_health():

    try:

        collections = client.get_collections()

        return {
            "database": "Qdrant",
            "status": "connected",
            "collection": COLLECTION_NAME,
            "collections": [
                collection.name
                for collection in collections.collections
            ]
        }

    except Exception as e:

        return {
            "database": "Qdrant",
            "status": "disconnected",
            "error": str(e)
        }
