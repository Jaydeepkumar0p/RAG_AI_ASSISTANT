import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database.mongodb import (
    check_database_connection,
)

from src.database.qdrant import (
    client,
    COLLECTION_NAME,
)

from src.routes.auth import (
    router as auth_router,
)

from src.routes.user import (
    router as user_router,
)

from src.routes.document import (
    router as document_router,
)

from src.routes.chat import (
    router as chat_router,
)

from src.routes.conversation import (
    router as conversation_router,
)

from src.services.rag_service import (
    create_collection,
)


# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="AI Study Assistant API",
    version="1.0.0",
)


# ======================================================
# CORS
# ======================================================

# Set this in Render:
#
# FRONTEND_URL=https://your-frontend.vercel.app
#
# Do not include the trailing slash.
#
frontend_url = (
    os.getenv(
        "FRONTEND_URL",
        "",
    )
    .strip()
    .rstrip("/")
)


# ------------------------------------------------------
# Local development origins
# ------------------------------------------------------

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "https://rag-frontend-ai-iota.vercel.app/",

    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]


# ------------------------------------------------------
# Add deployed frontend origin
# ------------------------------------------------------

if frontend_url:
    allowed_origins.append(
        frontend_url
    )


# ------------------------------------------------------
# Vercel preview deployments
# ------------------------------------------------------
#
# Allows:
#
# https://anything.vercel.app
# https://anything-anything.vercel.app
#
# If you do not use Vercel, this regex can be removed.
# ------------------------------------------------------

allowed_origin_regex = (
    r"^https://([a-zA-Z0-9-]+\.)?vercel\.app$"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=allowed_origins,

    allow_origin_regex=(
        allowed_origin_regex
    ),

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],

    max_age=600,
)


# ======================================================
# STARTUP
# ======================================================

@app.on_event("startup")
def startup_event():

    print(
        "Starting AI Study Assistant..."
    )


    # --------------------------------------------------
    # Ensure Qdrant collection
    # --------------------------------------------------

    try:

        create_collection()

        print(
            f"Qdrant collection ready: "
            f"{COLLECTION_NAME}"
        )

    except Exception as exc:

        print(
            "Qdrant collection initialization failed:",
            repr(exc),
        )

        # Do not prevent the API from starting.
        # Health endpoint will report the problem.


    # --------------------------------------------------
    # Ensure user_id index
    # --------------------------------------------------

    try:

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema="keyword",
        )

        print(
            "Qdrant user_id index ready"
        )

    except Exception as exc:

        print(
            "user_id index already exists "
            "or could not be created:",
            repr(exc),
        )


    # --------------------------------------------------
    # Ensure document_id index
    # --------------------------------------------------

    try:

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_id",
            field_schema="keyword",
        )

        print(
            "Qdrant document_id index ready"
        )

    except Exception as exc:

        print(
            "document_id index already exists "
            "or could not be created:",
            repr(exc),
        )


    print(
        "Application startup completed."
    )


# ======================================================
# ROUTES
# ======================================================

app.include_router(
    auth_router
)

app.include_router(
    user_router
)

app.include_router(
    document_router
)

app.include_router(
    chat_router
)

app.include_router(
    conversation_router
)


# ======================================================
# ROOT
# ======================================================

@app.get("/")
def root():

    return {
        "message": (
            "AI Study Assistant API is running"
        )
    }


# ======================================================
# MONGODB HEALTH
# ======================================================

@app.get("/health/db")
def database_health():

    connected = (
        check_database_connection()
    )

    if connected:

        return {
            "database": "MongoDB",
            "status": "connected",
        }

    return {
        "database": "MongoDB",
        "status": "disconnected",
    }


# ======================================================
# QDRANT HEALTH
# ======================================================

@app.get("/health/qdrant")
def qdrant_health():

    try:

        collections = (
            client.get_collections()
        )

        return {
            "database": "Qdrant",
            "status": "connected",
            "collection": COLLECTION_NAME,
            "collections": [
                collection.name
                for collection
                in collections.collections
            ],
        }

    except Exception as exc:

        return {
            "database": "Qdrant",
            "status": "disconnected",
            "error": str(exc),
        }
