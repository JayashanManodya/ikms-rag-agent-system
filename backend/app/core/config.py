"""Configuration management for the multi-agent RAG system.

This module uses Pydantic Settings to load and validate environment variables
for OpenAI models, Pinecone settings, and other system parameters.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# On Vercel (and other serverless platforms), there is no .env file —
# environment variables are injected directly by the platform.
# Passing env_file=None tells pydantic-settings to skip file lookup entirely.
_ENV_FILE = ".env" if os.path.exists(".env") else None

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    #OpenAI Configuration
    openai_api_key:str
    openai_model_name:str
    openai_embedding_model_name:str

    #Pinecone Configuration
    pinecone_api_key:str
    pinecone_index_name:str
    pinecone_environment: str = ""

    #Retrieval Configuration
    retrieval_k: int = 4

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

#Create a single instance of the settings(singleton pattern)
settings : Settings | None = None

def get_settings() -> Settings:
    """Get or create the application settings."""
    global settings
    if settings is None:
        settings = Settings()
    return settings