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

_fallback_lock = threading.Lock()
_fallback_counts: dict[str, int] = defaultdict(int)


def _record_fallback() -> int:
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    with _fallback_lock:
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

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["JSON_SORT_KEYS"] = False

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[Config.RATE_LIMIT],
    storage_uri="memory://",
)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Allow microphone when voice is enabled so getUserMedia() works in the browser.
    mic_policy = "microphone=*" if voice_enabled() else "microphone=()"
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/voice-status", methods=["GET"])
def voice_status():
    """Return the current voice feature status.

    Response:
        200 {"voice_enabled": false}   — when voice is disabled (default)
        200 {"voice_enabled": true}    — when voice is fully configured
    """
    return jsonify({"voice_enabled": voice_enabled()})

@app.route("/transcribe", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT)
def transcribe():
    """Transcribe uploaded audio to text using OpenAI Whisper.

    Request: multipart/form-data with field 'audio' (binary audio file).
    Response (200): {"text": "..."}
    Response (4xx/5xx): {"error": "..."}
    """
    if not voice_enabled():
        return jsonify({"error": "Voice features are not enabled"}), 403

    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file provided"}), 400

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
    if not voice_enabled():
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
