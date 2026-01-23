from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    DATABASE_URL: str
    # LLM Provider: "openai" ou "gemini"
    LLM_PROVIDER: Literal["openai", "gemini"] = "gemini"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Gemini (Google)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Storage: "local" ou "s3"
    STORAGE_BACKEND: Literal["local", "s3"] = "local"

    # AWS Settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: Optional[str] = None

    # SQS Settings
    SQS_QUEUE_URL: Optional[str] = None
    USE_SQS: bool = False

    class Config:
        env_file= ".env"

settings = Settings()