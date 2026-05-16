"""
FastAPI backend for Turkce Hoca.

Run locally:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CEFR_LEVELS, MODEL, retrieve_context
from content_intelligence import (
    CEFR_LEVELS as CEFR_LEVEL_NAMES,
    ExtractedContent,
    ExtractionError,
    build_study_prompt,
    extract_content,
    extract_text_from_file,
    extract_turkish_units,
    extract_vocabulary_items,
    infer_cefr_level,
    normalize_text,
)
from speech import list_macos_voices
from vocabulary_cards import (
    build_translation_lexicon,
    build_vocabulary_json_prompt,
    parse_vocabulary_cards,
)


REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class StudyUnit(BaseModel):
    text: str
    kind: str
    turkish_signal: bool


class VocabularyCardModel(BaseModel):
    turkish: str
    item_type: str
    translation: str
    cefr_level: str
    example_tr: str
    example_translation: str
    learner_note: str
    tts_word: str
    tts_sentence: str


class StudyResponse(BaseModel):
    source_type: str
    source_label: str
    inferred_level: str
    study_level: str
    target_language: str
    preview: str
    units: List[StudyUnit]
    vocabulary_cards: List[VocabularyCardModel]
    vocabulary_warning: str = ""
    note: str


class HealthResponse(BaseModel):
    ok: bool
    gemini_ready: bool
    model: str
    topics: int
    error: str = ""


GEMINI_STATE: Dict[str, Any] = {
    "available": False,
    "client": None,
    "error": "",
}


def read_env_api_key() -> str:
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return ""

    with env_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith("GEMINI_API_KEY="):
                continue
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def init_gemini() -> bool:
    if GEMINI_STATE["available"] and GEMINI_STATE["client"] is not None:
        return True

    api_key = read_env_api_key()
    if not api_key:
        GEMINI_STATE["error"] = "GEMINI_API_KEY is missing."
        return False

    try:
        from google import genai as modern_genai  # type: ignore

        GEMINI_STATE["client"] = modern_genai.Client(api_key=api_key)
        GEMINI_STATE["available"] = True
        GEMINI_STATE["error"] = ""
        return True
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        GEMINI_STATE["client"] = None
        GEMINI_STATE["available"] = False
        GEMINI_STATE["error"] = f"Could not initialize google-genai: {exc}"
        return False


def load_knowledge() -> List[Dict[str, Any]]:
    if not KNOWLEDGE_FILE.exists():
        subprocess.run([sys.executable, str(REPO_DIR / "dataset.py")], check=False)
    if not KNOWLEDGE_FILE.exists():
        return []
    with KNOWLEDGE_FILE.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("knowledge_base", [])


def generate_text(prompt: str) -> str:
    if not init_gemini():
        raise RuntimeError(GEMINI_STATE["error"] or "Gemini is not configured.")
    client = GEMINI_STATE["client"]
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return (getattr(response, "text", "") or "").strip()


async def ask_llm(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: generate_text(prompt))


def normalize_level(level: str, fallback: str = "A1") -> str:
    level = (level or fallback).upper()
    return level if level in CEFR_LEVEL_NAMES else fallback


def allowed_origins() -> List[str]:
    configured = os.environ.get("FRONTEND_ORIGIN") or os.environ.get("FRONTEND_ORIGINS", "")
    origins = DEFAULT_ALLOWED_ORIGINS[:]
    for origin in configured.split(","):
        origin = origin.strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return origins


async def extract_upload(upload: UploadFile, current_level: str) -> ExtractedContent:
    suffix = Path(upload.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        while chunk := await upload.read(1024 * 1024):
            handle.write(chunk)

    try:
        text, source_type = extract_text_from_file(temp_path)
        text = normalize_text(text)
        if not text:
            raise ExtractionError(f"No readable text found in {upload.filename or 'upload'}.")
        units = extract_turkish_units(text)
        inferred_level = infer_cefr_level(text, fallback=current_level)
        return ExtractedContent(
            source_type=source_type,
            source_label=upload.filename or "uploaded file",
            text=text,
            units=units,
            inferred_level=inferred_level,
        )
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


app = FastAPI(title="Turkce Hoca API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    knowledge = load_knowledge()
    gemini_ready = init_gemini()
    return HealthResponse(
        ok=bool(knowledge),
        gemini_ready=gemini_ready,
        model=MODEL,
        topics=len(knowledge),
        error="" if gemini_ready else GEMINI_STATE.get("error", ""),
    )


@app.get("/api/levels")
async def levels() -> Dict[str, Any]:
    return {"levels": CEFR_LEVELS}


@app.get("/api/voices")
async def voices(language: str = "auto") -> Dict[str, Any]:
    voice_list = list_macos_voices(language)
    return {
        "voices": [
            {
                "name": voice.name,
                "language": voice.language,
                "description": voice.description,
            }
            for voice in voice_list
        ]
    }


@app.post("/api/study", response_model=StudyResponse)
async def study(
    text: str = Form(""),
    level: str = Form("A1"),
    target_language: str = Form("English"),
    file: Optional[UploadFile] = File(None),
) -> StudyResponse:
    requested_level = normalize_level(level)
    try:
        if file and file.filename:
            extracted = await extract_upload(file, requested_level)
        else:
            extracted = extract_content(text, current_level=requested_level)
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    study_level = normalize_level(requested_level, extracted.inferred_level)
    knowledge = load_knowledge()
    context = retrieve_context(extracted.text, knowledge, study_level)
    vocabulary_items = extract_vocabulary_items(extracted.text)
    lexicon = build_translation_lexicon(knowledge)

    cards = []
    vocabulary_warning = ""
    if vocabulary_items:
        card_prompt = build_vocabulary_json_prompt(
            vocabulary_items,
            target_language,
            study_level,
        )
        try:
            card_response = await ask_llm(card_prompt)
            cards, vocabulary_warning = parse_vocabulary_cards(
                card_response,
                vocabulary_items,
                target_language,
                study_level,
                lexicon,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    prompt = build_study_prompt(extracted, target_language, study_level, context)

    try:
        note = await ask_llm(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not note:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

    return StudyResponse(
        source_type=extracted.source_type,
        source_label=extracted.source_label,
        inferred_level=extracted.inferred_level,
        study_level=study_level,
        target_language=target_language,
        preview=extracted.preview,
        units=[
            StudyUnit(
                text=unit.text,
                kind=unit.kind,
                turkish_signal=unit.turkish_signal,
            )
            for unit in extracted.units
        ],
        vocabulary_cards=[
            VocabularyCardModel(
                turkish=card.turkish,
                item_type=card.item_type,
                translation=card.translation,
                cefr_level=card.cefr_level,
                example_tr=card.example_tr,
                example_translation=card.example_translation,
                learner_note=card.learner_note,
                tts_word=card.tts_word,
                tts_sentence=card.tts_sentence,
            )
            for card in cards
        ],
        vocabulary_warning=vocabulary_warning,
        note=note,
    )
