from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)


class Settings(BaseSettings):

    # ==================================================
    # MONGODB
    # ==================================================

    MONGO_URI: str

    DATABASE_NAME: str = (
        "ai_study_assistant"
    )

    # ==================================================
    # JWT
    # ==================================================

    JWT_SECRET: str

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ==================================================
    # GROQ
    # ==================================================

    GROQ_API_KEY: str

    GROQ_MODEL: str

    # ==================================================
    # QDRANT
    # ==================================================

    QDRANT_URL: str = ""

    QDRANT_API_KEY: str = ""

    QDRANT_COLLECTION: str = (
        "my_documents"
    )

    # ==================================================
    # FILE UPLOAD
    # ==================================================

    UPLOAD_DIR: str = "uploads"

    # ==================================================
    # .ENV
    # ==================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()