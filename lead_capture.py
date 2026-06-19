"""
lead_capture.py — Conversational Lead Capture State Machine

Tracks per-session progress through collecting project details,
then emails the compiled lead to the company.

State machine stages (linear flow):
    idle → ask_name → ask_contact → ask_software → ask_work_details → confirm → done

Any stage except idle/done accepts "cancel" to reset cleanly to idle.

NOTE: Session state is stored in-memory (per-process). In a multi-worker
Gunicorn deployment, a user's request could land on a different worker
mid-flow and lose state. For production with >1 worker, replace the
_sessions dict with a Redis-backed store (store session as JSON, same
pattern as the rate limiter fix).
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

from config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# Session state store (in-memory, per-process)
# ============================================================================

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

SESSION_TTL_SECONDS = 1800  # 30 minutes of inactivity → session expires

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


def _cleanup_stale_sessions() -> None:
    """Remove sessions that have been inactive longer than SESSION_TTL_SECONDS.
    Must be called while holding _sessions_lock."""
    cutoff = _now() - SESSION_TTL_SECONDS
    stale = [sid for sid, s in _sessions.items() if s.get("last_active", 0) < cutoff]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.debug({"event": "sessions_cleaned", "count": len(stale)})


def _get_session(session_id: str) -> dict:
    """Return the session dict for session_id, creating it if absent.
    Updates last_active timestamp. Thread-safe."""
    with _sessions_lock:
        _cleanup_stale_sessions()
        if session_id not in _sessions:
            _sessions[session_id] = {
                "stage": STAGE_IDLE,
                "data": {},
                "last_active": _now(),
            }
        _sessions[session_id]["last_active"] = _now()
        return _sessions[session_id]


def _reset_session(session_id: str) -> None:
    """Reset session to idle state, clearing all collected data."""
    with _sessions_lock:
        _sessions[session_id] = {
            "stage": STAGE_IDLE,
            "data": {},
            "last_active": _now(),
        }


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
            return "No problem — let's redo this from the top. What's your name?"

        else:
            return (
                "Just to confirm — should I send these details to our team? "
                "Please reply **yes** or **no**."
            )

    # ------------------------------------------------------------------
    # DONE — session is complete; reset so a new flow can start fresh
    # ------------------------------------------------------------------
    if stage == STAGE_DONE:
        _reset_session(session_id)
        # Re-run as if the session is now idle so we don't silently eat the message
        return handle_lead_message(session_id, user_message)

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
    """Append the lead as a JSON line to leads_backup.jsonl.

    This is a safety net — if SMTP is not configured or fails, the
    data is still persisted locally and can be reviewed manually.
    File is excluded from git via .gitignore.
    """
    try:
        record = {**data, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open("leads_backup.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info({"event": "lead_backup_written"})
    except OSError as exc:
        logger.error({"event": "lead_backup_failed", "error": str(exc)})
