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
    WHISPER_LOCAL_BASE_URL = os.environ.get("WHISPER_LOCAL_BASE_URL", "")
    OPENROUTER_BASE_URL = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-31b-it:free")

    # Whitelist of models users may select via the chat API.
    # Add new models here to enable them; arbitrary model names are rejected.
    ALLOWED_MODELS: frozenset[str] = frozenset(
        {
            "google/gemma-4-31b-it:free",   # Gemma (default)
            "anthropic/claude-sonnet-4",    # Claude
            "openai/gpt-4o",               # GPT-4o
        }
    )

    # Redis URL for production rate limiting and session state.
    # When absent, both fall back to in-memory (single-worker / dev only).
    # Example: redis://localhost:6379/0  or  rediss://user:pass@host:6380/0
    REDIS_URL: str = os.environ.get("REDIS_URL", "")

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

    # ------------------------------------------------------------------
    # Voice — disabled by default; no logic fails if keys are absent
    # ------------------------------------------------------------------
    ENABLE_VOICE: bool = os.environ.get("ENABLE_VOICE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Placeholder API keys for future voice providers.
    # These are optional — the app starts fine without them.
    OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")          # OpenAI Whisper / TTS
    ELEVENLABS_API_KEY: str | None = os.environ.get("ELEVENLABS_API_KEY")  # ElevenLabs TTS
    DEEPGRAM_API_KEY: str | None = os.environ.get("DEEPGRAM_API_KEY")      # Deepgram STT

    # ------------------------------------------------------------------
    # Lead Capture / Email Notifications
    # SMTP credentials are optional — when absent, leads are still saved
    # to leads_backup.jsonl (excluded from git) as a safety net.
    # ------------------------------------------------------------------
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.environ.get("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.environ.get("SMTP_FROM_EMAIL", "")
    LEAD_NOTIFICATION_EMAIL: str = os.environ.get(
        "LEAD_NOTIFICATION_EMAIL", "muizznaveed@internetworks.io"
    )
