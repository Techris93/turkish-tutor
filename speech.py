"""
Language-aware text-to-speech helpers for Turkce Hoca.

The default implementation favors local playback through macOS `say`, with an
optional pyttsx3 fallback for other platforms. External premium providers can
be added later without changing tutor.py's command flow.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from content_intelligence import detect_language


DEFAULT_RATE = 175


class SpeechError(RuntimeError):
    """Raised when no usable speech engine is available."""


@dataclass(frozen=True)
class Voice:
    name: str
    language: str
    description: str = ""


@dataclass(frozen=True)
class SpeechOptions:
    language: str = "auto"
    voice: Optional[str] = None
    rate: int = DEFAULT_RATE


def normalize_language(language: str, text: str = "") -> str:
    language = (language or "auto").strip().lower()
    if language in {"auto", "detect"}:
        return detect_language(text)
    aliases = {
        "turkish": "tr",
        "tr-tr": "tr",
        "english": "en",
        "en-us": "en",
        "en-gb": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
    }
    return aliases.get(language, language.split("-")[0])


def _parse_say_voice_line(line: str) -> Optional[Voice]:
    # Example: "Yelda               tr_TR    # Merhaba, benim adım Yelda."
    match = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#\s*(.*)$", line.strip())
    if not match:
        return None
    return Voice(
        name=match.group(1).strip(),
        language=match.group(2).replace("_", "-").lower(),
        description=match.group(3).strip(),
    )


def list_macos_voices(language: str = "auto") -> List[Voice]:
    if not shutil.which("say"):
        return []
    result = subprocess.run(
        ["say", "-v", "?"],
        capture_output=True,
        text=True,
        check=False,
    )
    voices = [
        voice for line in result.stdout.splitlines()
        if (voice := _parse_say_voice_line(line))
    ]
    normalized = normalize_language(language)
    if normalized != "auto":
        voices = [
            voice for voice in voices
            if voice.language == normalized or voice.language.startswith(f"{normalized}-")
        ]
    return voices


def choose_voice(language: str, requested_voice: Optional[str] = None) -> Optional[str]:
    if requested_voice:
        return requested_voice
    voices = list_macos_voices(language)
    return voices[0].name if voices else None


def speak_with_macos_say(text: str, options: SpeechOptions) -> None:
    if not shutil.which("say"):
        raise SpeechError("macOS say command is not available.")

    language = normalize_language(options.language, text)
    voice = choose_voice(language, options.voice)
    rate = max(80, min(420, int(options.rate or DEFAULT_RATE)))

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temp_path = handle.name

    try:
        command = ["say", "-r", str(rate)]
        if voice:
            command.extend(["-v", voice])
        command.extend(["-f", temp_path])
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SpeechError("macOS say failed to play the text.")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def speak_with_pyttsx3(text: str, options: SpeechOptions) -> None:
    try:
        import pyttsx3  # type: ignore
    except ImportError as exc:
        raise SpeechError("Install pyttsx3 or run on macOS for local TTS.") from exc

    engine = pyttsx3.init()
    engine.setProperty("rate", max(80, min(420, int(options.rate or DEFAULT_RATE))))
    if options.voice:
        for voice in engine.getProperty("voices"):
            if options.voice.lower() in f"{voice.id} {voice.name}".lower():
                engine.setProperty("voice", voice.id)
                break
    engine.say(text)
    engine.runAndWait()


def speak(text: str, options: SpeechOptions | None = None) -> None:
    text = text.strip()
    if not text:
        raise SpeechError("No text provided for speech.")
    options = options or SpeechOptions()

    if platform.system() == "Darwin" and shutil.which("say"):
        speak_with_macos_say(text, options)
        return

    speak_with_pyttsx3(text, options)


def format_voice_list(language: str = "auto", limit: int = 20) -> str:
    voices = list_macos_voices(language)
    if not voices:
        return "No macOS voices found. On non-macOS systems, install pyttsx3 for basic playback."
    lines = []
    for voice in voices[:limit]:
        lines.append(f"{voice.name:24s} {voice.language:8s} {voice.description}")
    if len(voices) > limit:
        lines.append(f"... {len(voices) - limit} more")
    return "\n".join(lines)

