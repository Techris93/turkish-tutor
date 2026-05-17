from __future__ import annotations

import math
import os
import struct
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import httpx


class TTSConfigError(RuntimeError):
    pass


class TTSProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TTSRequest:
    text: str
    language: str
    voice: str | None = None
    speed: float = 1.0
    provider: str | None = None


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    media_type: str
    provider: str
    voice: str
    model: str


@dataclass(frozen=True)
class TTSProviderStatus:
    provider: str
    configured: bool
    auth_required: bool
    voices: list[str]
    model: str


OPENAI_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


def configured_provider_name() -> str:
    return os.environ.get("TTS_PROVIDER", "").strip().lower()


def tts_status() -> TTSProviderStatus:
    provider = configured_provider_name()
    if provider == "openai":
        return TTSProviderStatus(
            provider="openai",
            configured=bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            auth_required=True,
            voices=sorted(OPENAI_VOICES),
            model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        )
    if provider == "mock":
        return TTSProviderStatus(
            provider="mock",
            configured=True,
            auth_required=True,
            voices=["mock"],
            model="mock-tone",
        )
    return TTSProviderStatus(provider=provider or "none", configured=False, auth_required=True, voices=[], model="")


def select_openai_voice(language: str, requested_voice: str | None) -> str:
    if requested_voice:
        voice = requested_voice.strip().lower()
    elif language.lower().startswith("tr"):
        voice = os.environ.get("OPENAI_TTS_VOICE_TR", "nova").strip().lower()
    else:
        voice = os.environ.get("OPENAI_TTS_VOICE_DEFAULT", "alloy").strip().lower()
    if voice not in OPENAI_VOICES:
        raise TTSConfigError(f"Unsupported OpenAI TTS voice: {voice}.")
    return voice


def language_instructions(language: str) -> str:
    normalized = language.lower()
    if normalized.startswith("tr"):
        return "Pronounce the text naturally in Turkish with clear Turkish vowels and rhythm."
    return f"Pronounce the text naturally for language code {language}."


async def synthesize_openai(request: TTSRequest) -> TTSResult:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise TTSConfigError("OpenAI TTS is not configured. Set OPENAI_API_KEY.")
    voice = select_openai_voice(request.language, request.voice)
    model = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
    timeout = float(os.environ.get("OPENAI_TTS_TIMEOUT_SECONDS", "45"))
    payload = {
        "model": model,
        "voice": voice,
        "input": request.text,
        "response_format": "mp3",
        "speed": request.speed,
        "instructions": language_instructions(request.language),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise TTSProviderError(f"OpenAI TTS failed with status {response.status_code}.")
    if not response.content:
        raise TTSProviderError("OpenAI TTS returned empty audio.")
    return TTSResult(audio=response.content, media_type="audio/mpeg", provider="openai", voice=voice, model=model)


def sine_samples(duration_seconds: float = 0.18, sample_rate: int = 16_000) -> Iterable[int]:
    total = int(duration_seconds * sample_rate)
    for index in range(total):
        value = math.sin(2 * math.pi * 440 * index / sample_rate)
        yield int(value * 8000)


def mock_wav() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        frames = b"".join(struct.pack("<h", sample) for sample in sine_samples())
        handle.writeframes(frames)
    return buffer.getvalue()


async def synthesize_mock(request: TTSRequest) -> TTSResult:
    return TTSResult(audio=mock_wav(), media_type="audio/wav", provider="mock", voice="mock", model="mock-tone")


async def synthesize_tts(request: TTSRequest) -> TTSResult:
    provider = (request.provider or configured_provider_name()).strip().lower()
    if provider == "openai":
        return await synthesize_openai(request)
    if provider == "mock":
        return await synthesize_mock(request)
    if provider:
        raise TTSConfigError(f"Unsupported TTS provider: {provider}.")
    raise TTSConfigError("Generated audio is not configured. Set TTS_PROVIDER=openai and OPENAI_API_KEY.")
