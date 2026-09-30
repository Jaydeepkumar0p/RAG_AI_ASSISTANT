import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="AI RAG Assistant",
    version="1.0.0",
)


# ======================================================
# CORS
# ======================================================

frontend_url = os.getenv(
    "FRONTEND_URL",
    ""
).strip().rstrip("/")


allowed_origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]


# Add your deployed frontend exactly, if configured.
if frontend_url:
    allowed_origins.append(
        frontend_url
    )


# Optional Vercel preview deployments.
# Keep this only if your frontend is deployed on Vercel.
allowed_origin_regex = (
    r"^https://([a-zA-Z0-9-]+\.)?vercel\.app$"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=allowed_origins,

    allow_origin_regex=allowed_origin_regex,

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
# ROUTES
# ======================================================

from src.routes.auth import router as auth_router
from src.routes.user import router as user_router
from src.routes.document import router as document_router
from src.routes.chat import router as chat_router


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


# ======================================================
# ROOT
# ======================================================

@app.get("/")
async def root():
    return {
        "message": "API is running"
    }
