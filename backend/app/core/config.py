from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    falkordb_url: str = Field(default="redis://localhost:6379", description="FalkorDB Redis URL")
    groq_api_key: str = Field(default="", description="Groq API key")
    model: str = Field(default="openai/gpt-oss-120b", description="LLM model")
    temperature: float = Field(default=0.2, description="LLM temperature")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
