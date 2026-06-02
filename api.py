"""
FastAPI backend for Turkce Hoca.

Run locally:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth_storage import (
    SESSION_COOKIE_NAME,
    AuthSession,
    OAuthHandoff,
    OAuthState,
    PasswordResetToken,
    PracticeProgress as DBPracticeProgress,
    SavedLesson as DBSavedLesson,
    User,
    create_oauth_handoff,
    create_oauth_state,
    create_password_reset_token,
    create_session,
    find_user_by_email,
    get_db,
    hash_password,
    hash_token,
    init_db,
    isoformat,
    normalize_email,
    utcnow,
    verify_password,
)
from config import CEFR_LEVELS, MODEL, retrieve_context
from content_intelligence import (
    CEFR_LEVELS as CEFR_LEVEL_NAMES,
    ExtractedContent,
    ExtractionError,
    SUPPORTED_FILE_EXTENSIONS,
    build_study_prompt,
    extract_content,
    extract_text_from_file_details,
    extract_textbook_sections,
    extract_turkish_units,
    extract_vocabulary_items,
    infer_cefr_level,
    normalize_text,
)
from email_delivery import (
    EmailDeliveryError,
    build_password_reset_link,
    email_delivery_configured,
    send_password_reset_email,
)
from oauth_flow import (
    OAuthError,
    authorization_url,
    configured_providers,
    exchange_oauth_profile,
    oauth_error_redirect_url,
    oauth_success_redirect_url,
    provider_config,
)
from rate_limit import RateLimitRule, rate_limit
from speech import list_macos_voices
from tts_provider import (
    TTSConfigError,
    TTSProviderError,
    TTSRequest as ProviderTTSRequest,
    synthesize_tts,
    tts_status,
)
from textbook_breakdown import (
    build_textbook_breakdown_json_prompt,
    fallback_textbook_breakdown,
    parse_textbook_breakdown,
)
from vocabulary_cards import (
    build_translation_lexicon,
    build_vocabulary_json_prompt,
    fallback_card,
    parse_vocabulary_cards,
)


REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
DEFAULT_MAX_UPLOAD_BYTES = 80 * 1024 * 1024
DEFAULT_MAX_TEXT_INPUT_CHARS = 30_000
DEFAULT_LESSON_PAGE_LIMIT = 50
MAX_LESSON_PAGE_LIMIT = 100


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


class TextbookSectionModel(BaseModel):
    title: str
    section_type: str
    source_pages: str = ""
    level: str
    topic: str
    summary: str
    key_vocabulary: List[str] = Field(default_factory=list)
    grammar_focus: List[str] = Field(default_factory=list)
    translation: str
    practice: List[str] = Field(default_factory=list)


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
    textbook_sections: List[TextbookSectionModel] = Field(default_factory=list)
    textbook_warning: str = ""
    extraction_warning: str = ""
    note: str


class HealthResponse(BaseModel):
    ok: bool
    gemini_ready: bool
    model: str
    topics: int
    error: str = ""


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str


class SignupRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=256)
    name: str = Field("", max_length=120)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=256)


class AuthResponse(BaseModel):
    user: UserResponse
    session_token: Optional[str] = None


class PasswordResetRequest(BaseModel):
    email: str = Field(..., max_length=320)


class PasswordResetRequestResponse(BaseModel):
    message: str
    reset_token: Optional[str] = None
    email_delivery_configured: bool = False


class PasswordResetConfirmRequest(BaseModel):
    token: str = Field(..., min_length=16, max_length=256)
    password: str = Field(..., max_length=256)


class OAuthProvider(BaseModel):
    provider: str
    configured: bool
    authorization_url: Optional[str] = None


class OAuthConfigResponse(BaseModel):
    providers: List[OAuthProvider]


class OAuthRedeemRequest(BaseModel):
    handoff: str = Field(..., min_length=16, max_length=256)


class SavedLessonCreate(BaseModel):
    title: str = Field(..., max_length=180)
    result: StudyResponse


class SavedLessonUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=180)
    result: Optional[StudyResponse] = None


class SavedLessonResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    result: StudyResponse


class SavedLessonSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SavedLessonListResponse(BaseModel):
    lessons: List[SavedLessonSummary]
    limit: int
    offset: int
    total: int


class PracticeProgressUpdate(BaseModel):
    lesson_id: str = Field(..., min_length=1, max_length=80)
    progress: Dict[str, Any] = Field(default_factory=dict)


class PracticeProgressResponse(BaseModel):
    lesson_id: str
    progress: Dict[str, Any] = Field(default_factory=dict)
    exists: bool = False
    updated_at: str = ""


class OAuthRedeemResponse(BaseModel):
    user: UserResponse
    lessons: List[SavedLessonSummary]
    lessons_limit: int
    lessons_offset: int
    lessons_total: int
    session_token: str


class TTSConfigResponse(BaseModel):
    provider: str
    configured: bool
    auth_required: bool
    voices: List[str]
    model: str


class TTSAudioRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    language: str = Field("tr-TR", min_length=2, max_length=24)
    voice: Optional[str] = Field(None, max_length=80)
    speed: float = Field(1.0, ge=0.25, le=2.0)
    provider: Optional[str] = Field(None, max_length=40)


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
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    return (getattr(response, "text", "") or "").strip()


async def ask_llm(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: generate_text(prompt))


def provider_failure_summary(exc: RuntimeError) -> str:
    message = str(exc)
    if "503" in message or "UNAVAILABLE" in message or "high demand" in message.lower():
        return "Gemini is temporarily busy."
    if "GEMINI_API_KEY" in message or "not configured" in message:
        return "Gemini is not configured."
    return "The AI provider is temporarily unavailable."


def fallback_study_note(
    extracted: ExtractedContent,
    cards: List[Any],
    target_language: str,
    study_level: str,
    reason: str,
) -> str:
    """Create a useful study note when the AI provider is temporarily unavailable."""
    lines = [
        "AI study note is temporarily unavailable, so Türkçe Hoca used a deterministic fallback.",
        f"Study level: {study_level}. Target language: {target_language}.",
    ]
    if extracted.preview:
        lines.extend(["", "Extracted preview:", extracted.preview[:500]])
    if cards:
        lines.append("")
        lines.append("Vocabulary to review:")
        for card in cards[:16]:
            lines.append(f"- {card.turkish}: {card.translation}")
        lines.append("")
        lines.append("Quick practice:")
        for card in cards[:8]:
            lines.append(f"- {card.example_tr} = {card.example_translation}")
    else:
        lines.extend(
            [
                "",
                "Practice suggestion:",
                "- Read the extracted Turkish aloud once.",
                "- Underline verbs and suffixes.",
                "- Make three short sentences from the text.",
            ]
        )
    lines.extend(["", f"Provider note: {reason}"])
    return "\n".join(lines)


def normalize_level(level: str, fallback: str = "A1") -> str:
    level = (level or fallback).upper()
    return level if level in CEFR_LEVEL_NAMES else fallback


def allowed_origins() -> List[str]:
    configured = os.environ.get("FRONTEND_ORIGIN") or os.environ.get("FRONTEND_ORIGINS", "")
    origins = [] if os.environ.get("RENDER", "").lower() == "true" else DEFAULT_ALLOWED_ORIGINS[:]
    for origin in configured.split(","):
        origin = origin.strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def env_int(name: str, fallback: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def max_upload_bytes() -> int:
    return env_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


def max_text_input_chars() -> int:
    return env_int("MAX_TEXT_INPUT_CHARS", DEFAULT_MAX_TEXT_INPUT_CHARS)


def api_docs_enabled() -> bool:
    configured = os.environ.get("ENABLE_API_DOCS")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes", "on"}
    return os.environ.get("RENDER", "").lower() != "true"


def csrf_protection_enabled() -> bool:
    configured = os.environ.get("CSRF_PROTECTION_ENABLED")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes", "on"}
    return os.environ.get("RENDER", "").lower() == "true"


def origin_from_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def request_origin(request: Request) -> str:
    origin = request.headers.get("origin", "").strip().rstrip("/")
    if origin:
        return origin
    referer = request.headers.get("referer", "").strip()
    return origin_from_url(referer)


def request_url_origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}".rstrip("/")


def model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")


def user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        created_at=isoformat(user.created_at),
    )


def lesson_response(lesson: DBSavedLesson) -> SavedLessonResponse:
    return SavedLessonResponse(
        id=lesson.id,
        title=lesson.title,
        created_at=isoformat(lesson.created_at),
        updated_at=isoformat(lesson.updated_at),
        result=StudyResponse(**lesson.result),
    )


def lesson_summary_response(
    lesson_id: str,
    title: str,
    created_at: Any,
    updated_at: Any,
) -> SavedLessonSummary:
    return SavedLessonSummary(
        id=lesson_id,
        title=title,
        created_at=isoformat(created_at),
        updated_at=isoformat(updated_at),
    )


def lesson_summary_columns():
    return (
        DBSavedLesson.id,
        DBSavedLesson.title,
        DBSavedLesson.created_at,
        DBSavedLesson.updated_at,
    )


def count_user_lessons(db: Session, user_id: str) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(DBSavedLesson)
            .where(DBSavedLesson.user_id == user_id)
        )
        or 0
    )


def list_user_lesson_summaries(db: Session, user_id: str, limit: int, offset: int) -> SavedLessonListResponse:
    rows = db.execute(
        select(*lesson_summary_columns())
        .where(DBSavedLesson.user_id == user_id)
        .order_by(DBSavedLesson.updated_at.desc(), DBSavedLesson.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return SavedLessonListResponse(
        lessons=[lesson_summary_response(*row) for row in rows],
        limit=limit,
        offset=offset,
        total=count_user_lessons(db, user_id),
    )


def ensure_owned_lesson(db: Session, user_id: str, lesson_id: str) -> None:
    exists = db.scalar(
        select(DBSavedLesson.id).where(
            DBSavedLesson.id == lesson_id,
            DBSavedLesson.user_id == user_id,
        )
    )
    if exists is None:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")


def practice_progress_response(progress: DBPracticeProgress, exists: bool = True) -> PracticeProgressResponse:
    return PracticeProgressResponse(
        lesson_id=progress.lesson_id,
        progress=progress.progress or {},
        exists=exists,
        updated_at=isoformat(progress.updated_at),
    )


def cookie_secure() -> bool:
    configured = os.environ.get("AUTH_COOKIE_SECURE")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes", "on"}
    return os.environ.get("RENDER", "").lower() == "true"


def cookie_samesite() -> str:
    value = os.environ.get("AUTH_COOKIE_SAMESITE", "lax").lower()
    return value if value in {"lax", "strict", "none"} else "lax"


def set_session_cookie(response: Response, token: str) -> None:
    max_age = int(os.environ.get("AUTH_SESSION_DAYS", "30")) * 24 * 60 * 60
    same_site = cookie_samesite()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=cookie_secure() or same_site == "none",
        samesite=same_site,  # type: ignore[arg-type]
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    same_site = cookie_samesite()
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=cookie_secure() or same_site == "none",
        samesite=same_site,  # type: ignore[arg-type]
    )


def is_expired(expires_at) -> bool:  # type: ignore[no-untyped-def]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=utcnow().tzinfo)
    return expires_at <= utcnow()


def public_api_url(request: Request, path: str) -> str:
    base = os.environ.get("PUBLIC_API_URL", "").strip().rstrip("/")
    if base:
        return f"{base}{path}"
    return str(request.url_for("health")).removesuffix("/api/health") + path


def redirect_with_params(url: str, params: Dict[str, str]) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def request_session_token(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE_NAME) or request.headers.get("x-session-token")


def authenticate_session_token(token: str, db: Session) -> User:
    auth_session = db.get(AuthSession, hash_token(token))
    if auth_session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    if is_expired(auth_session.expires_at):
        db.delete(auth_session)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired.")

    user = db.get(User, auth_session.user_id)
    if user is None:
        db.delete(auth_session)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return user


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request_session_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return authenticate_session_token(token, db)


def optional_current_user(request: Request, db: Session) -> Optional[User]:
    token = request_session_token(request)
    if not token:
        return None
    try:
        return authenticate_session_token(token, db)
    except HTTPException:
        return None


async def extract_upload(upload: UploadFile, current_level: str) -> ExtractedContent:
    source_label = Path(upload.filename or "uploaded file").name
    suffix = Path(source_label).suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
        raise ExtractionError(f"Unsupported input file type '{suffix or '(none)'}'. Supported: {supported}.")

    total_bytes = 0
    byte_limit = max_upload_bytes()
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            while chunk := await upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > byte_limit:
                    raise ExtractionError(f"Uploaded file is too large. Limit is {byte_limit // (1024 * 1024)} MB.")
                handle.write(chunk)

        text, source_type, extraction_warning = extract_text_from_file_details(temp_path)
        text = normalize_text(text)
        if not text:
            raise ExtractionError(f"No readable text found in {source_label}.")
        units = extract_turkish_units(text)
        inferred_level = infer_cefr_level(text, fallback=current_level)
        textbook_sections = extract_textbook_sections(text) if source_type in {"pdf", "document", "text-file"} else []
        return ExtractedContent(
            source_type=source_type,
            source_label=source_label,
            text=text,
            units=units,
            inferred_level=inferred_level,
            textbook_sections=textbook_sections,
            extraction_warning=extraction_warning,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


app = FastAPI(
    title="Turkce Hoca API",
    version="1.0.0",
    docs_url="/docs" if api_docs_enabled() else None,
    redoc_url="/redoc" if api_docs_enabled() else None,
    openapi_url="/openapi.json" if api_docs_enabled() else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if not api_docs_enabled():
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")
    if request.url.path.startswith(("/api/auth", "/api/lessons", "/api/study", "/api/tts/audio")):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.middleware("http")
async def csrf_origin_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
    unsafe_method = request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
    cookie_session = request.cookies.get(SESSION_COOKIE_NAME)
    if csrf_protection_enabled() and unsafe_method and cookie_session:
        origin = request_origin(request)
        trusted_origins = set(allowed_origins())
        trusted_origins.add(request_url_origin(request))
        if not origin or origin not in trusted_origins:
            return JSONResponse(status_code=403, content={"detail": "Cross-site request rejected."})
    return await call_next(request)


@app.on_event("startup")
async def startup() -> None:
    init_db()


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


@app.get("/api/tts/config", response_model=TTSConfigResponse)
async def tts_config() -> TTSConfigResponse:
    status_payload = tts_status()
    return TTSConfigResponse(
        provider=status_payload.provider,
        configured=status_payload.configured,
        auth_required=status_payload.auth_required,
        voices=status_payload.voices,
        model=status_payload.model,
    )


@app.post("/api/tts/audio")
async def tts_audio(
    payload: TTSAudioRequest,
    request: Request,
    user: User = Depends(current_user),
) -> Response:
    rate_limit(request, "tts", RateLimitRule(120, 3600), user.id)
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for generated audio.")
    try:
        audio = await synthesize_tts(
            ProviderTTSRequest(
                text=text,
                language=payload.language.strip(),
                voice=payload.voice.strip() if payload.voice else None,
                speed=payload.speed,
                provider=payload.provider.strip().lower() if payload.provider else None,
            )
        )
    except TTSConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TTSProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=audio.audio,
        media_type=audio.media_type,
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Provider": audio.provider,
            "X-TTS-Voice": audio.voice,
            "X-TTS-Model": audio.model,
        },
    )


@app.post("/api/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    rate_limit(request, "signup", RateLimitRule(5, 3600))
    email = validate_email(payload.email)
    validate_password(payload.password)
    name = payload.name.strip() or email.split("@", 1)[0]

    user = User(email=email, name=name, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    db.refresh(user)
    token = create_session(db, user)
    set_session_cookie(response, token)
    return AuthResponse(user=user_response(user), session_token=token)


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    rate_limit(request, "login", RateLimitRule(10, 300))
    email = validate_email(payload.email)
    user = find_user_by_email(db, email)
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(db, user)
    set_session_cookie(response, token)
    return AuthResponse(user=user_response(user), session_token=token)


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> Dict[str, bool]:
    token = request_session_token(request)
    if token:
        auth_session = db.get(AuthSession, hash_token(token))
        if auth_session is not None:
            db.delete(auth_session)
            db.commit()
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/auth/me", response_model=AuthResponse)
async def me(user: User = Depends(current_user)) -> AuthResponse:
    return AuthResponse(user=user_response(user))


@app.post("/api/auth/password-reset/request", response_model=PasswordResetRequestResponse)
async def request_password_reset(
    payload: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PasswordResetRequestResponse:
    rate_limit(request, "password_reset", RateLimitRule(5, 3600))
    email = validate_email(payload.email)
    user = find_user_by_email(db, email)
    reset_token = None
    if user is not None:
        token = create_password_reset_token(db, user)
        if email_delivery_configured():
            try:
                send_password_reset_email(user.email, token)
            except EmailDeliveryError:
                pass
        if os.environ.get("PASSWORD_RESET_RETURN_TOKEN", "").lower() in {"1", "true", "yes"}:
            reset_token = token

    email_configured = email_delivery_configured()
    message = (
        "If an account exists, a reset token was created. Configure SMTP_HOST and SMTP_FROM_EMAIL to email it."
        if not email_configured
        else "If an account exists, password reset instructions will be sent."
    )
    return PasswordResetRequestResponse(
        message=message,
        reset_token=reset_token,
        email_delivery_configured=email_configured,
    )


@app.post("/api/auth/password-reset/confirm")
async def confirm_password_reset(
    payload: PasswordResetConfirmRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Dict[str, bool]:
    rate_limit(request, "password_reset_confirm", RateLimitRule(10, 3600))
    validate_password(payload.password)
    reset = db.get(PasswordResetToken, hash_token(payload.token))
    if reset is None or reset.used_at is not None or is_expired(reset.expires_at):
        raise HTTPException(status_code=400, detail="Invalid or expired password reset token.")
    user = db.get(User, reset.user_id)
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid or expired password reset token.")

    user.password_hash = hash_password(payload.password)
    reset.used_at = utcnow()
    db.query(AuthSession).filter(AuthSession.user_id == user.id).delete()
    db.commit()
    return {"ok": True}


@app.get("/api/auth/oauth/config", response_model=OAuthConfigResponse)
async def oauth_config(request: Request) -> OAuthConfigResponse:
    providers = configured_providers()
    return OAuthConfigResponse(
        providers=[
            OAuthProvider(
                provider="google",
                configured=providers["google"],
                authorization_url=public_api_url(request, "/api/auth/oauth/google/start")
                if providers["google"]
                else None,
            ),
            OAuthProvider(
                provider="github",
                configured=providers["github"],
                authorization_url=public_api_url(request, "/api/auth/oauth/github/start")
                if providers["github"]
                else None,
            ),
        ]
    )


@app.get("/api/auth/oauth/{provider}/start", name="oauth_start")
async def oauth_start(
    provider: str,
    request: Request,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    provider = provider.lower()
    rate_limit(request, f"oauth_{provider}_start", RateLimitRule(20, 3600))
    if provider_config(provider) is None:
        raise HTTPException(status_code=404, detail="OAuth provider is not configured.")
    state = create_oauth_state(db, provider)
    return RedirectResponse(authorization_url(provider, state), status_code=302)


@app.get("/api/auth/oauth/{provider}/callback", name="oauth_callback")
async def oauth_callback(
    provider: str,
    request: Request,
    response: Response,
    code: str = "",
    state: str = "",
    error: str = "",
    db: Session = Depends(get_db),
) -> RedirectResponse:
    provider = provider.lower()
    rate_limit(request, f"oauth_{provider}_callback", RateLimitRule(40, 3600))
    if error:
        return RedirectResponse(oauth_error_redirect_url("provider_error"), status_code=302)
    if provider_config(provider) is None:
        raise HTTPException(status_code=404, detail="OAuth provider is not configured.")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Invalid OAuth callback.")

    saved_state = db.get(OAuthState, hash_token(state))
    if saved_state is None or saved_state.provider != provider or is_expired(saved_state.expires_at):
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")

    db.delete(saved_state)
    db.commit()

    try:
        profile = await exchange_oauth_profile(provider, code)
    except OAuthError:
        return RedirectResponse(oauth_error_redirect_url("profile_error"), status_code=302)

    signed_in_user = optional_current_user(request, db)
    if signed_in_user is not None and signed_in_user.email != profile.email:
        return RedirectResponse(oauth_error_redirect_url("account_mismatch"), status_code=302)

    user = signed_in_user or find_user_by_email(db, profile.email)
    if user is None:
        user = User(
            email=profile.email,
            name=profile.name or profile.email.split("@", 1)[0],
            password_hash=hash_password(secrets.token_urlsafe(32)),
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            user = find_user_by_email(db, profile.email)
            if user is None:
                return RedirectResponse(oauth_error_redirect_url("account_error"), status_code=302)
        else:
            db.refresh(user)

    handoff = create_oauth_handoff(db, user)
    token = create_session(db, user)
    redirect = RedirectResponse(
        redirect_with_params(oauth_success_redirect_url(), {"handoff": handoff}),
        status_code=302,
    )
    set_session_cookie(redirect, token)
    return redirect


@app.post("/api/auth/oauth/redeem", response_model=OAuthRedeemResponse)
async def oauth_redeem(
    payload: OAuthRedeemRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> OAuthRedeemResponse:
    rate_limit(request, "oauth_redeem", RateLimitRule(20, 3600))
    handoff = db.get(OAuthHandoff, hash_token(payload.handoff))
    if handoff is None or handoff.used_at is not None or is_expired(handoff.expires_at):
        raise HTTPException(status_code=400, detail="OAuth sign-in could not be verified.")

    user = db.get(User, handoff.user_id)
    if user is None:
        raise HTTPException(status_code=400, detail="OAuth sign-in could not be verified.")

    handoff.used_at = utcnow()
    db.commit()
    token = create_session(db, user)
    set_session_cookie(response, token)
    lesson_page = list_user_lesson_summaries(db, user.id, DEFAULT_LESSON_PAGE_LIMIT, 0)
    return OAuthRedeemResponse(
        user=user_response(user),
        lessons=lesson_page.lessons,
        lessons_limit=lesson_page.limit,
        lessons_offset=lesson_page.offset,
        lessons_total=lesson_page.total,
        session_token=token,
    )


@app.get("/api/lessons", response_model=SavedLessonListResponse)
async def list_lessons(
    limit: int = Query(DEFAULT_LESSON_PAGE_LIMIT, ge=1, le=MAX_LESSON_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonListResponse:
    return list_user_lesson_summaries(db, user.id, limit, offset)


@app.post("/api/lessons", response_model=SavedLessonResponse, status_code=status.HTTP_201_CREATED)
async def create_lesson(
    payload: SavedLessonCreate,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonResponse:
    rate_limit(request, "lesson_write", RateLimitRule(120, 3600), user.id)
    title = payload.title.strip() or "Turkish lesson"
    lesson = DBSavedLesson(user_id=user.id, title=title, result=model_dump(payload.result))
    db.add(lesson)
    db.commit()
    db.refresh(lesson)
    return lesson_response(lesson)


@app.get("/api/lessons/{lesson_id}", response_model=SavedLessonResponse)
async def get_lesson(
    lesson_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonResponse:
    lesson = db.scalar(
        select(DBSavedLesson).where(
            DBSavedLesson.id == lesson_id,
            DBSavedLesson.user_id == user.id,
        )
    )
    if lesson is None:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    return lesson_response(lesson)


@app.patch("/api/lessons/{lesson_id}", response_model=SavedLessonResponse)
async def update_lesson(
    lesson_id: str,
    payload: SavedLessonUpdate,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonResponse:
    rate_limit(request, "lesson_write", RateLimitRule(120, 3600), user.id)
    lesson = db.scalar(
        select(DBSavedLesson).where(
            DBSavedLesson.id == lesson_id,
            DBSavedLesson.user_id == user.id,
        )
    )
    if lesson is None:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")

    if payload.title is not None:
        lesson.title = payload.title.strip() or lesson.title
    if payload.result is not None:
        lesson.result = model_dump(payload.result)
    lesson.updated_at = utcnow()
    db.commit()
    db.refresh(lesson)
    return lesson_response(lesson)


@app.delete("/api/lessons/{lesson_id}")
async def delete_lesson(
    lesson_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict[str, bool]:
    rate_limit(request, "lesson_write", RateLimitRule(120, 3600), user.id)
    lesson = db.scalar(
        select(DBSavedLesson).where(
            DBSavedLesson.id == lesson_id,
            DBSavedLesson.user_id == user.id,
        )
    )
    if lesson is None:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    db.delete(lesson)
    db.commit()
    return {"ok": True}


@app.get("/api/practice/progress", response_model=PracticeProgressResponse)
async def get_practice_progress(
    lesson_id: str = Query(..., min_length=1, max_length=80),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> PracticeProgressResponse:
    ensure_owned_lesson(db, user.id, lesson_id)
    progress = db.scalar(
        select(DBPracticeProgress).where(
            DBPracticeProgress.user_id == user.id,
            DBPracticeProgress.lesson_id == lesson_id,
        )
    )
    if progress is None:
        return PracticeProgressResponse(lesson_id=lesson_id, progress={}, exists=False, updated_at="")
    return practice_progress_response(progress)


@app.put("/api/practice/progress", response_model=PracticeProgressResponse)
async def put_practice_progress(
    payload: PracticeProgressUpdate,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> PracticeProgressResponse:
    rate_limit(request, "practice_write", RateLimitRule(240, 3600), user.id)
    ensure_owned_lesson(db, user.id, payload.lesson_id)
    serialized = json.dumps(payload.progress, ensure_ascii=False)
    if len(serialized) > 50_000:
        raise HTTPException(status_code=413, detail="Practice progress is too large.")

    progress = db.scalar(
        select(DBPracticeProgress).where(
            DBPracticeProgress.user_id == user.id,
            DBPracticeProgress.lesson_id == payload.lesson_id,
        )
    )
    if progress is None:
        progress = DBPracticeProgress(user_id=user.id, lesson_id=payload.lesson_id, progress=payload.progress)
        db.add(progress)
    else:
        progress.progress = payload.progress
        progress.updated_at = utcnow()
    db.commit()
    db.refresh(progress)
    return practice_progress_response(progress)


@app.post("/api/study", response_model=StudyResponse)
async def study(
    request: Request,
    text: str = Form(""),
    level: str = Form("A1"),
    target_language: str = Form("English"),
    file: Optional[UploadFile] = File(None),
) -> StudyResponse:
    rate_limit(request, "study", RateLimitRule(30, 3600))
    requested_level = normalize_level(level)
    target_language = target_language.strip()[:80] or "English"
    if len(text) > max_text_input_chars():
        raise HTTPException(status_code=413, detail=f"Text input is too large. Limit is {max_text_input_chars()} characters.")
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
            failure_summary = provider_failure_summary(exc)
            cards = [
                fallback_card(item, target_language, study_level, lexicon)
                for item in vocabulary_items
            ]
            vocabulary_warning = f"AI vocabulary generation unavailable; deterministic fallback used. {failure_summary}"

    textbook_sections = []
    textbook_warning = ""
    if extracted.textbook_sections:
        textbook_prompt = build_textbook_breakdown_json_prompt(
            extracted,
            target_language,
            study_level,
        )
        try:
            textbook_response = await ask_llm(textbook_prompt)
            textbook_sections, textbook_warning = parse_textbook_breakdown(
                textbook_response,
                extracted.textbook_sections,
                target_language,
                study_level,
            )
        except RuntimeError as exc:
            failure_summary = provider_failure_summary(exc)
            textbook_sections = fallback_textbook_breakdown(
                extracted.textbook_sections,
                target_language,
                study_level,
            )
            textbook_warning = f"AI textbook breakdown unavailable; deterministic fallback used. {failure_summary}"

    prompt = build_study_prompt(extracted, target_language, study_level, context)

    try:
        note = await ask_llm(prompt)
    except RuntimeError as exc:
        note = fallback_study_note(
            extracted,
            cards,
            target_language,
            study_level,
            provider_failure_summary(exc),
        )

    if not note:
        note = fallback_study_note(
            extracted,
            cards,
            target_language,
            study_level,
            "Gemini returned an empty response.",
        )

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
        textbook_sections=[
            TextbookSectionModel(
                title=section.title,
                section_type=section.section_type,
                source_pages=section.source_pages,
                level=section.level,
                topic=section.topic,
                summary=section.summary,
                key_vocabulary=section.key_vocabulary,
                grammar_focus=section.grammar_focus,
                translation=section.translation,
                practice=section.practice,
            )
            for section in textbook_sections
        ],
        textbook_warning=textbook_warning,
        extraction_warning=extracted.extraction_warning,
        note=note,
    )
