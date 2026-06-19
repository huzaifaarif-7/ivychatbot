import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from config import Config
from fallback_kb import get_fallback_response
from internetworks import (
    call_openrouter_with_timeout,
    get_calendly_response,
    is_meeting_request,
    sanitize_user_message,
)
from voice_service import voice_enabled, transcribe_audio, generate_voice_response
from lead_capture import (
    handle_lead_message,
    is_in_lead_capture,
    should_start_lead_capture,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed MIME types for audio uploads (guard against non-audio payloads).
# ---------------------------------------------------------------------------
_ALLOWED_AUDIO_TYPES = frozenset({
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "application/octet-stream",  # some browsers send this as fallback
})

# ---------------------------------------------------------------------------
# Fallback counter (with bounded memory — prune entries older than 2 hours).
# ---------------------------------------------------------------------------
_fallback_lock = threading.Lock()
_fallback_counts: dict[str, int] = defaultdict(int)


def _record_fallback() -> int:
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    with _fallback_lock:
        # Prune keys older than the current and previous hour to bound memory.
        keep = {
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H"),
            (datetime.now(timezone.utc).replace(hour=max(0, datetime.now(timezone.utc).hour - 1))).strftime("%Y-%m-%dT%H"),
        }
        stale = [k for k in _fallback_counts if k not in keep]
        for k in stale:
            del _fallback_counts[k]
        _fallback_counts[hour_key] += 1
        return _fallback_counts[hour_key]


def _log_fallback_triggered(error: Exception) -> None:
    count_this_hour = _record_fallback()
    logger.warning(
        {
            "event": "fallback_triggered",
            "error_type": type(error).__name__,
            "error_msg": str(error)[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback_count_this_hour": count_this_hour,
            "source": "fallback",
        }
    )


# ---------------------------------------------------------------------------
# Cache voice_enabled() once at startup — it reads config, never needs I/O.
# ---------------------------------------------------------------------------
_VOICE_ENABLED: bool = voice_enabled()

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["JSON_SORT_KEYS"] = False

# ---------------------------------------------------------------------------
# Rate limiter — uses Redis in production, falls back to memory in dev.
# ---------------------------------------------------------------------------
_limiter_storage = (
    f"redis://{Config.REDIS_URL.split('://', 1)[-1]}"
    if Config.REDIS_URL
    else "memory://"
)
if Config.REDIS_URL:
    logger.info({"event": "rate_limiter_backend", "backend": "redis"})
else:
    logger.warning({
        "event": "rate_limiter_backend",
        "backend": "memory",
        "warning": "In-memory rate limiting is per-process. Set REDIS_URL for multi-worker safety.",
    })

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[Config.RATE_LIMIT],
    storage_uri=Config.REDIS_URL if Config.REDIS_URL else "memory://",
)


# ---------------------------------------------------------------------------
# CSRF / Origin check — rejects cross-origin POST requests.
# This is a lightweight same-origin guard; add flask-wtf tokens if you add
# authenticated sessions.
# ---------------------------------------------------------------------------
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@app.before_request
def check_origin():
    """Reject POST/PUT/PATCH/DELETE requests that originate from a different
    site. Only applied when the server knows its own host (non-debug or
    when a HOST env var is set).
    """
    if request.method in _SAFE_METHODS:
        return  # GET requests are always allowed

    # In debug mode with no explicit host configured, skip the check so
    # local curl / Postman testing still works.
    if Config.FLASK_DEBUG:
        return

    origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
    if not origin:
        # No Origin/Referer header — could be a same-origin request from
        # an older browser. Allow it (strict mode would block, but that
        # would break some legitimate same-origin flows).
        return

    server_host = request.host  # e.g. "yourdomain.com" or "yourdomain.com:443"
    if server_host not in origin:
        logger.warning({
            "event": "csrf_origin_rejected",
            "origin": origin[:200],
            "host": server_host,
            "path": request.path,
        })
        return jsonify({"error": "Forbidden"}), 403


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # (self) not * — only our own page may request mic access, never a third-party iframe.
    mic_policy = "microphone=(self)" if _VOICE_ENABLED else "microphone=()"
    response.headers["Permissions-Policy"] = f"geolocation=(), {mic_policy}, camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://assets.calendly.com; "
        "style-src 'self' 'unsafe-inline' https://assets.calendly.com; "
        "img-src 'self' data: https://*.calendly.com https://assets.calendly.com; "
        "connect-src 'self' https://calendly.com https://*.calendly.com; "
        "frame-src https://calendly.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    if not Config.FLASK_DEBUG:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    return jsonify({"error": error.description}), error.code


@app.errorhandler(Exception)
def handle_unexpected_exception(error: Exception):
    logger.exception("Unhandled application error")
    return jsonify({"error": "An internal error occurred."}), 500


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe for load balancers and uptime monitors.

    Returns 200 with a minimal JSON body. No auth required.
    Does NOT exercise downstream dependencies (OpenRouter, Redis) — use
    a separate readiness probe for that.
    """
    return jsonify({"status": "ok", "voice_enabled": _VOICE_ENABLED})


@app.route("/robots.txt", methods=["GET"])
def robots():
    """Serve robots.txt from the static directory at the root URL path."""
    return send_file(
        os.path.join(app.static_folder, "robots.txt"),
        mimetype="text/plain",
    )


@app.route("/voice-status", methods=["GET"])
def voice_status():
    """Return the current voice feature status.

    Response:
        200 {"voice_enabled": false}   — when voice is disabled (default)
        200 {"voice_enabled": true}    — when voice is fully configured
    """
    return jsonify({"voice_enabled": _VOICE_ENABLED})


@app.route("/transcribe", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT)
def transcribe():
    """Transcribe uploaded audio to text using OpenAI Whisper.

    Request: multipart/form-data with field 'audio' (binary audio file).
    Response (200): {"text": "..."}
    Response (4xx/5xx): {"error": "..."}
    """
    if not _VOICE_ENABLED:
        return jsonify({"error": "Voice features are not enabled"}), 403

    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file provided"}), 400

    # Validate MIME type before spending an API call.
    content_type = (audio_file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_AUDIO_TYPES:
        logger.warning({
            "event": "transcribe_rejected",
            "reason": "invalid_content_type",
            "content_type": content_type,
        })
        return jsonify({"error": "Unsupported audio format."}), 415

    # Basic size guard — reject anything over 25 MB (Whisper's own limit).
    audio_file.seek(0, 2)
    size = audio_file.tell()
    audio_file.seek(0)
    if size > 25 * 1024 * 1024:
        return jsonify({"error": "Audio file too large (max 25 MB)"}), 400

    result = transcribe_audio(audio_file.read(), audio_file.mimetype)

    if result["success"]:
        return jsonify({"text": result["transcript"]})

    logger.warning({"event": "transcription_failed", "error": result["error"]})
    return jsonify({"error": "Could not transcribe audio. Please try again or type your message."}), 500


@app.route("/speak", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT)
def speak():
    """Convert text to speech using OpenAI TTS and return MP3 audio.

    Request: JSON {"text": "..."}
    Response (200): audio/mpeg binary stream
    Response (4xx/5xx): {"error": "..."}
    """
    if not _VOICE_ENABLED:
        return jsonify({"error": "Voice features are not enabled"}), 403

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400
    if len(text) > 4000:
        text = text[:4000]

    result = generate_voice_response(text)

    if not result.get("success"):
        return jsonify({"error": result.get("error", "Speech generation failed")}), 500

    return send_file(
        BytesIO(result["audio_bytes"]),
        mimetype="audio/mpeg",
        as_attachment=False,
    )


@app.route("/models", methods=["GET"])
def list_models():
    """Return the allowed model list with human-readable labels.

    The frontend uses this to populate the model selector and to validate
    user selections. The backend whitelist (Config.ALLOWED_MODELS) is the
    single source of truth — the frontend never hardcodes model IDs.

    Response shape:
        {
          "default": "google/gemma-4-31b-it:free",
          "models": [
            {"id": "google/gemma-4-31b-it:free", "label": "Gemma"},
            ...
          ]
        }
    """
    # Ordered display list — order matters for the UI dropdown.
    ordered = [
        {"id": "google/gemma-4-31b-it:free", "label": "Gemma"},
        {"id": "anthropic/claude-sonnet-4",  "label": "Claude"},
        {"id": "openai/gpt-4o",              "label": "GPT-4o"},
    ]
    # Only surface models that are in the whitelist (guard against stale list).
    available = [m for m in ordered if m["id"] in Config.ALLOWED_MODELS]
    return jsonify({"default": Config.LLM_MODEL, "models": available})


@app.route("/chat", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT)
def chat():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    user_message = payload.get("message")
    if user_message is None:
        return jsonify({"error": "Missing required field: message."}), 400

    try:
        user_message = sanitize_user_message(user_message)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # --- Session ID (required for lead-capture state tracking) --------------
    session_id = payload.get("session_id")
    if not session_id or not isinstance(session_id, str):
        return jsonify({"error": "Missing required field: session_id."}), 400
    # Clamp to a safe length to prevent memory abuse
    session_id = session_id[:128]
    # -------------------------------------------------------------------------

    # --- Model selection (optional field) ------------------------------------
    # When "model" is absent or None, call_openrouter_with_timeout uses the
    # default Config.LLM_MODEL (Gemma) — existing behaviour is fully preserved.
    requested_model = payload.get("model")  # None when not supplied
    if requested_model is not None:
        if not isinstance(requested_model, str):
            return jsonify({"error": "Field 'model' must be a string."}), 400
        if requested_model not in Config.ALLOWED_MODELS:
            return (
                jsonify(
                    {
                        "error": "Unsupported model. Choose one of: "
                        + ", ".join(sorted(Config.ALLOWED_MODELS))
                    }
                ),
                400,
            )
    # -------------------------------------------------------------------------

    # --- Lead capture (runs before meeting / AI checks) ----------------------
    # Once a session is in the lead-capture flow, ALL messages are handled by
    # the state machine — we do not want a meeting keyword mid-flow to abort
    # the conversation and redirect to Calendly.
    if is_in_lead_capture(session_id) or should_start_lead_capture(user_message):
        reply = handle_lead_message(session_id, user_message)
        logger.info({"event": "chat_response", "source": "lead_capture", "session_id": session_id})
        return jsonify({"response": reply})
    # -------------------------------------------------------------------------

    if is_meeting_request(user_message):
        return jsonify(get_calendly_response())

    try:
        bot_response = call_openrouter_with_timeout(user_message, model=requested_model)
        logger.info({"event": "chat_response", "source": "ai", "model": requested_model or Config.LLM_MODEL})
    except Exception as exc:
        _log_fallback_triggered(exc)
        bot_response = get_fallback_response(user_message)
        logger.info({"event": "chat_response", "source": "fallback"})

    if isinstance(bot_response, dict):
        return jsonify(bot_response)

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(
        debug=Config.FLASK_DEBUG,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
    )
