"""Application settings loaded from .env."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfigSettings(BaseSettings):
    """Runtime configuration values for the application."""

    AZURE_ENDPOINT: str = ""
    AZURE_DEPLOYMENT_NAME: str = ""
    AZURE_API_VERSION: str = ""

    AGENTS_LLM_MAX_TOKENS: int = 4096
    AGENTS_LLM_TEMPERATURE: float = 0.1
    AGENTS_LLM_TOP_P: float = 0.9

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "default.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def logging_config(self) -> dict[str, Any]:
        """Build a stdlib logging dictConfig payload from settings."""
        level = self.LOG_LEVEL.upper()
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": level,
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "default",
                    "level": level,
                    "filename": self.LOG_FILE,
                },
            },
            "root": {"handlers": ["console", "file"], "level": level},
        }


settings = AppConfigSettings()
