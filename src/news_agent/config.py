"""Configuration loaded from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # News source API keys (at least one required)
    newsapi_key: str = ""
    finnhub_key: str = ""
    alphavantage_key: str = ""

    # FRED API key (for economic indicators — free at https://fred.stlouisfed.org/docs/api/api_key.html)
    fred_key: str = ""

    # Hugging Face
    hf_token: str = ""
    hf_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    llm_backend: str = "api"  # "api" or "local"

    # Agent behaviour
    watch_interval_seconds: int = 300  # 5 min between polls in watch mode

    @property
    def has_any_news_source(self) -> bool:
        return any([self.newsapi_key, self.finnhub_key, self.alphavantage_key])


def load_settings() -> Settings:
    return Settings()
