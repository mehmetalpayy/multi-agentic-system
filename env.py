"""Environment-backed secret configuration loaded from .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretSettings(BaseSettings):
    """Loads secret values from the repository root .env file."""

    AZURE_OPENAI_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


secrets = SecretSettings()
