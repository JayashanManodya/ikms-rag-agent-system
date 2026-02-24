"""Configuration management for the multi-agent RAG system.

This module uses Pydantic Settings to load and validate environment variables
for OpenAI models, Pinecone settings, and other system parameters.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

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
        env_file=".env",
        env_file_encoding="utf-8",
        env_file_required=False,  # Don't crash if .env is absent (e.g. on Vercel)
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