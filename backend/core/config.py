"""Centralized application settings."""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "inference_jobs.db"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Distributed Memory Grid (DMG) API Gateway"
    app_description: str = (
        "Real-time distributed LLM inference engine control plane and streaming gateway."
    )
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]
    database_url: str = f"sqlite:///{_DEFAULT_DATABASE_PATH.as_posix()}"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, log_level: str) -> str:
        normalized = log_level.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )
        return normalized

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        if not origins:
            raise ValueError("CORS_ORIGINS must contain at least one non-empty origin")

        normalized_origins: list[str] = []
        for value in origins:
            origin = value.strip()
            if not origin:
                raise ValueError(
                    "CORS_ORIGINS must contain at least one non-empty origin"
                )
            if "*" in origin:
                raise ValueError("CORS_ORIGINS must not contain wildcard origins")
            if any(character.isspace() for character in origin):
                raise ValueError(
                    "CORS_ORIGINS entries must be valid HTTP or HTTPS origins"
                )

            try:
                parsed = urlsplit(origin)
                port = parsed.port
            except ValueError as error:
                raise ValueError(
                    "CORS_ORIGINS entries must be valid HTTP or HTTPS origins"
                ) from error

            if port is None and parsed.netloc.endswith(":"):
                raise ValueError(
                    "CORS_ORIGINS entries must be valid HTTP or HTTPS origins"
                )
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "CORS_ORIGINS entries must be valid HTTP or HTTPS origins"
                )

            normalized_origins.append(origin)

        return normalized_origins


settings = Settings()
