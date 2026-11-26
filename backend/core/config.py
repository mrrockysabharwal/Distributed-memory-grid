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
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, secret: str) -> str:
        normalized = secret.strip()
        insecure_placeholders = {
            "change-me",
            "changeme",
            "change_me",
            "example",
            "secret",
            "password",
            "jwt-secret",
            "jwt_secret",
        }

        if not normalized:
            raise ValueError("JWT_SECRET_KEY must be supplied through the environment")
        if len(normalized) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 characters")
        if normalized.lower().startswith(("change_me", "change-me", "changeme")):
            raise ValueError("JWT_SECRET_KEY must not use an example or placeholder")
        if normalized.lower() in insecure_placeholders:
            raise ValueError("JWT_SECRET_KEY is insecure")

        return normalized

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, algorithm: str) -> str:
        normalized = algorithm.strip().upper()
        if normalized != "HS256":
            raise ValueError("JWT_ALGORITHM must be HS256")
        return normalized

    @field_validator("access_token_expire_minutes")
    @classmethod
    def validate_access_token_expire_minutes(cls, minutes: int) -> int:
        if minutes <= 0:
            raise ValueError(
                "ACCESS_TOKEN_EXPIRE_MINUTES must be greater than zero"
            )
        if minutes > 10080:
            raise ValueError(
                "ACCESS_TOKEN_EXPIRE_MINUTES must not exceed 10080 minutes"
            )
        return minutes

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
