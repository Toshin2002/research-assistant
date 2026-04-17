from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str
    tavily_api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    max_iterations: int = 5
    database_url: str = "sqlite+aiosqlite:///./research_agent.db"


settings = Settings()
