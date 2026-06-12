import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request
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
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
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

    if is_meeting_request(user_message):
        return jsonify(get_calendly_response())

    try:
        bot_response = call_openrouter_with_timeout(user_message)
        logger.info({"event": "chat_response", "source": "ai"})
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
