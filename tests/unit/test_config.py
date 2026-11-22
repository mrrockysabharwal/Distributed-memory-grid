"""Tests for centralized backend settings."""

import pytest
from pydantic import ValidationError

from backend.core.config import Settings


SETTING_ENV_VARS = (
    "APP_NAME",
    "APP_DESCRIPTION",
    "APP_VERSION",
    "HOST",
    "PORT",
    "DEBUG",
    "LOG_LEVEL",
    "CORS_ORIGINS",
)


def test_default_settings(monkeypatch):
    for variable in SETTING_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "Distributed Memory Grid (DMG) API Gateway"
    assert settings.app_description == (
        "Real-time distributed LLM inference engine control plane and streaming gateway."
    )
    assert settings.app_version == "0.1.0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.debug is True
    assert settings.log_level == "INFO"
    assert settings.cors_origins == ["http://localhost:5173"]


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Test Gateway")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("CORS_ORIGINS", '["https://example.test"]')

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test Gateway"
    assert settings.port == 9000
    assert settings.cors_origins == ["https://example.test"]


@pytest.mark.parametrize(
    "log_level",
    ["debug", "info", "warning", "error", "critical"],
)
def test_lowercase_log_levels_are_normalized(log_level):
    settings = Settings(_env_file=None, log_level=log_level)

    assert settings.log_level == log_level.upper()


@pytest.mark.parametrize("log_level", ["TRACE", "verbose", ""])
def test_invalid_log_levels_are_rejected(log_level):
    with pytest.raises(ValidationError, match="LOG_LEVEL must be one of"):
        Settings(_env_file=None, log_level=log_level)


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("  http://localhost:5173  ", "http://localhost:5173"),
        ("https://example.test:8443", "https://example.test:8443"),
    ],
)
def test_valid_http_origins_are_preserved_and_trimmed(origin, expected):
    settings = Settings(_env_file=None, cors_origins=[origin])

    assert settings.cors_origins == [expected]


@pytest.mark.parametrize(
    "origin",
    [
        "localhost:5173",
        "http://",
        "https:///missing-host",
        "ftp://example.test",
        "http://localhost:99999",
        "http://localhost:",
    ],
)
def test_invalid_http_origins_are_rejected(origin):
    with pytest.raises(ValidationError, match="valid HTTP or HTTPS origins"):
        Settings(_env_file=None, cors_origins=[origin])


def test_wildcard_cors_origin_is_rejected():
    with pytest.raises(ValidationError, match="must not contain wildcard"):
        Settings(_env_file=None, cors_origins=["https://*.example.test"])


def test_empty_cors_origin_strings_are_rejected():
    with pytest.raises(ValidationError, match="at least one non-empty origin"):
        Settings(_env_file=None, cors_origins=["  "])


def test_empty_cors_origins_are_rejected():
    with pytest.raises(ValidationError, match="at least one non-empty origin"):
        Settings(_env_file=None, cors_origins=[])
