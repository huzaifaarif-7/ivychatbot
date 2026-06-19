"""
lead_capture.py — Conversational Lead Capture State Machine

Tracks per-session progress through collecting project details,
then emails the compiled lead to the company.

State machine stages (linear flow):
    idle → ask_name → ask_contact → ask_software → ask_work_details → confirm → done

Any stage except idle/done accepts "cancel" to reset cleanly to idle.

SESSION STORAGE
---------------
Tries Redis first (if Config.REDIS_URL is set) for multi-worker safety.
Falls back to an in-memory dict when Redis is unavailable.
In-memory mode is safe for single-worker deployments (dev / simple prod)
but sessions will not survive worker restarts and will not be shared across
multiple Gunicorn workers.
"""

import json
import logging
import re
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Absolute path for the lead backup file — safe regardless of cwd.
# ---------------------------------------------------------------------------
_BACKUP_PATH = Path(__file__).parent / "data" / "leads_backup.jsonl"

# ============================================================================
# Session storage — Redis preferred, in-memory fallback
# ============================================================================

SESSION_TTL_SECONDS = 1800  # 30 minutes of inactivity → session expires

_redis_client = None  # populated below if Redis is available


def _init_redis():
    """Try to connect to Redis. Returns the client or None on failure."""
    if not Config.REDIS_URL:
        return None
    try:
        import redis  # type: ignore
        client = redis.from_url(Config.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        logger.info({"event": "session_store_backend", "backend": "redis"})
        return client
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning({
            "event": "session_store_backend",
            "backend": "memory_fallback",
            "reason": str(exc)[:200],
        })
        return None


_redis_client = _init_redis()

# In-memory fallback store (used when Redis is unavailable)
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

STAGE_IDLE = "idle"
STAGE_ASK_NAME = "ask_name"
STAGE_ASK_CONTACT = "ask_contact"
STAGE_ASK_SOFTWARE = "ask_software"
STAGE_ASK_WORK_DETAILS = "ask_work_details"
STAGE_CONFIRM = "confirm"
STAGE_DONE = "done"

# ---------------------------------------------------------------------------
# Trigger / control patterns
# ---------------------------------------------------------------------------

_TRIGGER_PATTERNS = re.compile(
    r"\b(start a project|work with you|want to build|want to make|"
    r"hire (you|internetworks)|collaborate|partner with|develop my|"
    r"build my (project|app|software|website|system)|"
    r"i have a project|need (a developer|help building)|"
    r"work on my project|make my project)\b",
    re.IGNORECASE,
)

_CANCEL_PATTERNS = re.compile(
    r"\b(cancel|never mind|nevermind|stop|forget it|not now|"
    r"start over|restart)\b",
    re.IGNORECASE,
)

_CONFIRM_YES = re.compile(
    r"\b(yes|yeah|yep|confirm|correct|send it|go ahead)\b",
    re.IGNORECASE,
)
_CONFIRM_NO = re.compile(
    r"\b(no|nope|wrong|incorrect|change|edit)\b",
    re.IGNORECASE,
)


# ============================================================================
# Internal helpers
# ============================================================================


def _now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

_REDIS_PREFIX = "ivy:session:"


def _redis_get(session_id: str) -> dict | None:
    """Fetch session dict from Redis. Returns None on miss or error."""
    if not _redis_client:
        return None
    try:
        raw = _redis_client.get(f"{_REDIS_PREFIX}{session_id}")
        return json.loads(raw) if raw else None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning({"event": "redis_get_error", "error": str(exc)[:200]})
        return None


def _redis_set(session_id: str, session: dict) -> bool:
    """Write session dict to Redis with TTL. Returns True on success."""
    if not _redis_client:
        return False
    try:
        _redis_client.setex(
            f"{_REDIS_PREFIX}{session_id}",
            SESSION_TTL_SECONDS,
            json.dumps(session),
        )
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning({"event": "redis_set_error", "error": str(exc)[:200]})
        return False


def _redis_delete(session_id: str) -> None:
    if not _redis_client:
        return
    try:
        _redis_client.delete(f"{_REDIS_PREFIX}{session_id}")
    except Exception:  # pylint: disable=broad-except
        pass


# ---------------------------------------------------------------------------
# In-memory helpers
# ---------------------------------------------------------------------------


def _mem_cleanup_stale() -> None:
    """Remove expired in-memory sessions. Must be called while holding lock."""
    cutoff = _now() - SESSION_TTL_SECONDS
    stale = [sid for sid, s in _sessions.items() if s.get("last_active", 0) < cutoff]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.debug({"event": "sessions_cleaned", "count": len(stale)})


# ---------------------------------------------------------------------------
# Unified session API
# ---------------------------------------------------------------------------


def _get_session(session_id: str) -> dict:
    """Return the session dict for session_id, creating it if absent.
    Updates last_active. Tries Redis first, falls back to in-memory.
    """
    # --- Redis path ---
    session = _redis_get(session_id)
    if session is not None:
        session["last_active"] = _now()
        _redis_set(session_id, session)
        return session

    # --- In-memory path ---
    with _sessions_lock:
        _mem_cleanup_stale()
        if session_id not in _sessions:
            _sessions[session_id] = {
                "stage": STAGE_IDLE,
                "data": {},
                "last_active": _now(),
            }
        _sessions[session_id]["last_active"] = _now()
        # Sync to Redis if available (handles the case where in-memory has it
        # but Redis missed it due to a transient error on the last write).
        if _redis_client:
            _redis_set(session_id, _sessions[session_id])
        return _sessions[session_id]


def _save_session(session_id: str, session: dict) -> None:
    """Persist session changes to whichever backend is active."""
    if not _redis_set(session_id, session):
        # Redis unavailable — write/update in-memory store.
        with _sessions_lock:
            _sessions[session_id] = session


def _reset_session(session_id: str) -> None:
    """Reset session to idle state, clearing all collected data."""
    blank = {"stage": STAGE_IDLE, "data": {}, "last_active": _now()}
    if not _redis_set(session_id, blank):
        with _sessions_lock:
            _sessions[session_id] = blank


def _sanitize_for_email(value: str) -> str:
    """Strip newlines and control characters to prevent SMTP header injection.
    Truncates at 500 chars."""
    return re.sub(r"[\r\n\x00-\x1f]+", " ", value).strip()[:500]


# ============================================================================
# Public API
# ============================================================================


def is_in_lead_capture(session_id: str) -> bool:
    """Return True when this session is mid-flow (not idle or done)."""
    session = _get_session(session_id)
    return session["stage"] not in (STAGE_IDLE, STAGE_DONE)


def should_start_lead_capture(user_message: str) -> bool:
    """Return True when a normal message matches the lead-capture trigger phrases."""
    return bool(_TRIGGER_PATTERNS.search(user_message))


def handle_lead_message(session_id: str, user_message: str) -> str:
    """Advance the state machine by one turn and return the bot's reply.

    Call this INSTEAD of the normal AI chat path whenever
    ``is_in_lead_capture()`` or ``should_start_lead_capture()`` is True.
    The function is safe to call re-entrantly from different threads as
    long as each call uses a distinct session_id.
    """
    session = _get_session(session_id)
    stage = session["stage"]
    data = session["data"]

    # ------------------------------------------------------------------
    # Global cancel — accepted at any active stage
    # ------------------------------------------------------------------
    if stage not in (STAGE_IDLE, STAGE_DONE) and _CANCEL_PATTERNS.search(user_message):
        _reset_session(session_id)
        logger.info({"event": "lead_capture_cancelled", "session_id": session_id})
        return (
            "No problem — I've cancelled that. "
            "Let me know anytime if you'd like to share your project details "
            "with our team!"
        )

    # ------------------------------------------------------------------
    # IDLE → entry point
    # ------------------------------------------------------------------
    if stage == STAGE_IDLE:
        session["stage"] = STAGE_ASK_NAME
        _save_session(session_id, session)
        logger.info({"event": "lead_capture_started", "session_id": session_id})
        return (
            "That's great to hear! I'd love to connect you with our team "
            "so we can help bring your project to life. Let's grab a few "
            "quick details first.\n\n"
            "What's your name?"
        )

    # ------------------------------------------------------------------
    # ASK_NAME → collect name, advance to ASK_CONTACT
    # ------------------------------------------------------------------
    if stage == STAGE_ASK_NAME:
        name = _sanitize_for_email(user_message)
        if len(name) < 2:
            return "Could you share your name again? Even just your first name works."
        data["name"] = name
        session["stage"] = STAGE_ASK_CONTACT
        _save_session(session_id, session)
        return (
            f"Nice to meet you, {name}! "
            f"What's the best phone number or email address to reach you at?"
        )

    # ------------------------------------------------------------------
    # ASK_CONTACT → collect contact, advance to ASK_SOFTWARE
    # ------------------------------------------------------------------
    if stage == STAGE_ASK_CONTACT:
        contact = _sanitize_for_email(user_message)
        if len(contact) < 5:
            return (
                "That doesn't look quite right — "
                "could you share your phone number or email again?"
            )
        data["contact"] = contact
        session["stage"] = STAGE_ASK_SOFTWARE
        _save_session(session_id, session)
        return (
            "Got it! Now, what kind of software or technology are you "
            "looking to build? "
            "(e.g. web app, mobile app, AI integration, automation, etc.)"
        )

    # ------------------------------------------------------------------
    # ASK_SOFTWARE → collect software type, advance to ASK_WORK_DETAILS
    # ------------------------------------------------------------------
    if stage == STAGE_ASK_SOFTWARE:
        software = _sanitize_for_email(user_message)
        if len(software) < 2:
            return "Could you tell me a bit more about what you're looking to build?"
        data["software"] = software
        session["stage"] = STAGE_ASK_WORK_DETAILS
        _save_session(session_id, session)
        return (
            "Great! Last thing — can you describe the project in a bit more "
            "detail? Things like timeline, scope, or any specific requirements "
            "are very helpful."
        )

    # ------------------------------------------------------------------
    # ASK_WORK_DETAILS → collect details, advance to CONFIRM
    # ------------------------------------------------------------------
    if stage == STAGE_ASK_WORK_DETAILS:
        work_details = _sanitize_for_email(user_message)
        if len(work_details) < 2:
            return "Could you share a bit more detail about the project?"
        data["work_details"] = work_details
        session["stage"] = STAGE_CONFIRM
        _save_session(session_id, session)
        return (
            "Here's what I've got:\n\n"
            f"• **Name:** {data['name']}\n"
            f"• **Contact:** {data['contact']}\n"
            f"• **Software / Tech:** {data['software']}\n"
            f"• **Project details:** {data['work_details']}\n\n"
            "Shall I send this to our team so they can reach out to you? "
            "(yes / no)"
        )

    # ------------------------------------------------------------------
    # CONFIRM → send or restart
    # ------------------------------------------------------------------
    if stage == STAGE_CONFIRM:
        if _CONFIRM_YES.search(user_message):
            success = _send_lead_email(data)
            session["stage"] = STAGE_DONE
            _save_session(session_id, session)
            logger.info(
                {
                    "event": "lead_capture_completed",
                    "session_id": session_id,
                    "email_sent": success,
                }
            )
            if success:
                return (
                    "Perfect — I've sent your details to our team! "
                    "Someone from Internetworks will reach out to you shortly. "
                    "Thanks for your interest in working with us! 🎉"
                )
            else:
                # Email failed, but we don't expose that detail to the user.
                # Data has already been written to the JSONL backup.
                return (
                    "Thanks! I've recorded your details and our team will "
                    "follow up with you soon."
                )

        elif _CONFIRM_NO.search(user_message):
            # User wants to redo — restart from name
            session["stage"] = STAGE_ASK_NAME
            data.clear()
            _save_session(session_id, session)
            return "No problem — let's redo this from the top. What's your name?"

        else:
            return (
                "Just to confirm — should I send these details to our team? "
                "Please reply **yes** or **no**."
            )

    # ------------------------------------------------------------------
    # DONE — FIX: do NOT recurse. Reset and return a neutral reply.
    # Recursing caused IVY to restart the lead capture flow for any
    # follow-up message sent immediately after completion.
    # ------------------------------------------------------------------
    if stage == STAGE_DONE:
        _reset_session(session_id)
        logger.info({"event": "lead_capture_post_done_reset", "session_id": session_id})
        return (
            "Is there anything else I can help you with regarding Internetworks?"
        )

    # ------------------------------------------------------------------
    # Safety net — should never be reached
    # ------------------------------------------------------------------
    logger.error(
        {"event": "lead_capture_unknown_stage", "stage": stage, "session_id": session_id}
    )
    _reset_session(session_id)
    return (
        "Something went wrong with the form flow — I've reset it. "
        "Please try again."
    )


# ============================================================================
# Email sending
# ============================================================================


def _send_lead_email(data: dict) -> bool:
    """Send collected lead data via SMTP.

    Always writes a JSONL backup first, regardless of email outcome.
    Returns True on successful email delivery, False on any failure.
    """
    # Write backup unconditionally — email may still fail
    _backup_lead_to_file(data)

    if not (Config.SMTP_HOST and Config.SMTP_USERNAME and Config.SMTP_PASSWORD):
        logger.warning(
            {
                "event": "lead_email_skipped",
                "reason": "SMTP not configured — lead saved to backup file only",
            }
        )
        return False

    try:
        subject = f"New Project Lead from IVY: {data.get('name', 'Unknown')}"
        body = (
            "New lead captured via the IVY chatbot\n"
            f"Timestamp : {datetime.now(timezone.utc).isoformat()}\n"
            "─" * 40 + "\n\n"
            f"Name          : {data.get('name', '')}\n"
            f"Contact       : {data.get('contact', '')}\n"
            f"Software/Tech : {data.get('software', '')}\n"
            f"Project Details:\n{data.get('work_details', '')}\n"
        )

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = formataddr(("IVY Chatbot", Config.SMTP_FROM_EMAIL))
        msg["To"] = Config.LEAD_NOTIFICATION_EMAIL

        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
            server.sendmail(
                Config.SMTP_FROM_EMAIL,
                [Config.LEAD_NOTIFICATION_EMAIL],
                msg.as_string(),
            )

        logger.info(
            {
                "event": "lead_email_sent",
                "to": Config.LEAD_NOTIFICATION_EMAIL,
                "name": data.get("name", ""),
            }
        )
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            {"event": "lead_email_failed", "reason": "SMTP authentication failed"}
        )
        return False
    except smtplib.SMTPException as exc:
        logger.error({"event": "lead_email_failed", "error": str(exc)})
        return False
    except OSError as exc:
        logger.error(
            {
                "event": "lead_email_failed",
                "reason": "Network/socket error",
                "error": str(exc),
            }
        )
        return False


def _backup_lead_to_file(data: dict) -> None:
    """Append the lead as a JSON line to the backup file.

    Uses an absolute path anchored to the project root so the file
    location is predictable regardless of the process working directory.
    File is excluded from git via .gitignore.
    """
    try:
        _BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {**data, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(_BACKUP_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info({"event": "lead_backup_written", "path": str(_BACKUP_PATH)})
    except OSError as exc:
        logger.error({"event": "lead_backup_failed", "error": str(exc)})
