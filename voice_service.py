"""
voice_service.py — Voice Implementation (OpenAI Whisper STT + OpenAI TTS)

Providers used:
  - Speech-To-Text: OpenAI Whisper (model: whisper-1)
  - Text-To-Speech: OpenAI TTS    (model: tts-1, voice: nova)

To enable voice:
  1. Set ENABLE_VOICE=true in your .env
  2. Set OPENAI_API_KEY in your .env
  3. Restart the server — /voice-status will return {"voice_enabled": true}

Note: ELEVENLABS_API_KEY and DEEPGRAM_API_KEY are reserved for future
alternate provider support but are not currently used.
"""

import io
import logging

from config import Config

logger = logging.getLogger(__name__)

# Maximum text length accepted by OpenAI TTS (hard API limit)
_TTS_MAX_CHARS = 4096


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def voice_enabled() -> bool:
    """Return True only when voice is fully configured and enabled.

    Requires both ENABLE_VOICE=true and a valid OPENAI_API_KEY.
    """
    return Config.ENABLE_VOICE and bool(Config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Speech-To-Text  (OpenAI Whisper)
# ---------------------------------------------------------------------------


def transcribe_audio(audio_data: bytes, mime_type: str = "audio/webm") -> dict:
    """Convert raw audio bytes to text using OpenAI Whisper.

    Args:
        audio_data: Raw audio bytes captured from the browser microphone.
        mime_type:  MIME type of the audio (e.g. "audio/webm", "audio/wav").

    Returns:
        A dict with the following keys:
          - success    (bool):       Whether transcription succeeded.
          - transcript (str | None): The transcribed text, or None on failure.
          - error      (str | None): Human-readable error message, or None on success.
    """
    if not voice_enabled():
        logger.debug("transcribe_audio called but voice is not configured.")
        return {
            "success": False,
            "transcript": None,
            "error": "Voice is not configured. Set ENABLE_VOICE=true and OPENAI_API_KEY.",
        }

    if not audio_data:
        return {
            "success": False,
            "transcript": None,
            "error": "No audio data received.",
        }

    # Build a filename extension that matches the mime type so Whisper can
    # infer the codec.  Default to .webm (the format the browser sends).
    _mime_to_ext = {
        "audio/webm": "recording.webm",
        "audio/ogg":  "recording.ogg",
        "audio/wav":  "recording.wav",
        "audio/mpeg": "recording.mp3",
        "audio/mp4":  "recording.mp4",
    }
    filename = _mime_to_ext.get(mime_type, "recording.webm")

    logger.info(
        {
            "event": "transcription_attempt",
            "audio_bytes": len(audio_data),
            "mime_type": mime_type,
        }
    )

    try:
        import openai  # local import keeps startup fast when voice is disabled

        # client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        client = openai.OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.WHISPER_LOCAL_BASE_URL or None,
        )
        # Whisper requires a file-like object with a .name attribute.
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename

        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

        transcript = response.text.strip()
        logger.info(
            {
                "event": "transcription_success",
                "transcript_length": len(transcript),
            }
        )
        return {"success": True, "transcript": transcript, "error": None}

    except openai.AuthenticationError:
        logger.error({"event": "transcription_failed", "reason": "invalid_api_key"})
        return {
            "success": False,
            "transcript": None,
            "error": "Invalid OpenAI API key. Check OPENAI_API_KEY in your .env.",
        }

    except openai.BadRequestError as exc:
        logger.warning(
            {"event": "transcription_failed", "reason": "bad_request", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "transcript": None,
            "error": "Audio format not supported or file is corrupt.",
        }

    except openai.APITimeoutError:
        logger.warning({"event": "transcription_failed", "reason": "timeout"})
        return {
            "success": False,
            "transcript": None,
            "error": "Transcription timed out. Please try again.",
        }

    except openai.APIError as exc:
        logger.error(
            {"event": "transcription_failed", "reason": "api_error", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "transcript": None,
            "error": "OpenAI API error during transcription.",
        }

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(
            {"event": "transcription_failed", "reason": "unexpected", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "transcript": None,
            "error": "An unexpected error occurred during transcription.",
        }


# ---------------------------------------------------------------------------
# Text-To-Speech  (OpenAI TTS)
# ---------------------------------------------------------------------------


def generate_voice_response(text: str, voice_id: str | None = None) -> dict:
    """Convert a text string to audio bytes using OpenAI TTS.

    Args:
        text:     The text to synthesise into speech.
        voice_id: Optional voice name (e.g. "alloy", "nova", "shimmer").
                  Defaults to "nova" if not provided.

    Returns:
        A dict with the following keys:
          - success     (bool):       Whether synthesis succeeded.
          - audio_bytes (bytes|None): Raw MP3 audio data, or None on failure.
          - mime_type   (str|None):   "audio/mpeg" on success, None on failure.
          - error       (str|None):   Human-readable error message, or None on success.
    """
    if not voice_enabled():
        logger.debug("generate_voice_response called but voice is not configured.")
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "Voice is not configured. Set ENABLE_VOICE=true and OPENAI_API_KEY.",
        }

    if not text or not text.strip():
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "No text provided for speech synthesis.",
        }

    # Enforce OpenAI TTS character limit — truncate rather than error so the
    # user still gets a (partial) audio response.
    if len(text) > _TTS_MAX_CHARS:
        logger.warning(
            {
                "event": "tts_text_truncated",
                "original_length": len(text),
                "truncated_to": _TTS_MAX_CHARS,
            }
        )
        text = text[:_TTS_MAX_CHARS]

    voice = voice_id or "nova"  # nova sounds natural and clear
    logger.info({"event": "tts_attempt", "text_length": len(text), "voice": voice})

    try:
        import openai  # local import keeps startup fast when voice is disabled

        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )

        audio_bytes = response.read()
        logger.info({"event": "tts_success", "audio_bytes": len(audio_bytes)})
        return {
            "success": True,
            "audio_bytes": audio_bytes,
            "mime_type": "audio/mpeg",
            "error": None,
        }

    except openai.AuthenticationError:
        logger.error({"event": "tts_failed", "reason": "invalid_api_key"})
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "Invalid OpenAI API key. Check OPENAI_API_KEY in your .env.",
        }

    except openai.BadRequestError as exc:
        logger.warning(
            {"event": "tts_failed", "reason": "bad_request", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "Text-to-speech request was rejected by the API.",
        }

    except openai.APITimeoutError:
        logger.warning({"event": "tts_failed", "reason": "timeout"})
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "Speech synthesis timed out. Please try again.",
        }

    except openai.APIError as exc:
        logger.error(
            {"event": "tts_failed", "reason": "api_error", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "OpenAI API error during speech synthesis.",
        }

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(
            {"event": "tts_failed", "reason": "unexpected", "detail": str(exc)[:200]}
        )
        return {
            "success": False,
            "audio_bytes": None,
            "mime_type": None,
            "error": "An unexpected error occurred during speech synthesis.",
        }
