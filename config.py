import os
import secrets

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    OPENROUTER_API_KEY = _require_env("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-31b-it:free")
    OPENROUTER_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "8"))
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "2000"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
    RATE_LIMIT = os.environ.get("RATE_LIMIT", "20 per minute")
    CALENDLY_URL = os.environ.get(
        "CALENDLY_URL", "https://calendly.com/muizznaveed-internetworks/30min"
    )
