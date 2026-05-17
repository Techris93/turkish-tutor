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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth_storage import (
    SESSION_COOKIE_NAME,
    AuthSession,
    PasswordResetToken,
    SavedLesson as DBSavedLesson,
    User,
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


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user: UserResponse


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetRequestResponse(BaseModel):
    message: str
    reset_token: Optional[str] = None
    email_delivery_configured: bool = False


class PasswordResetConfirmRequest(BaseModel):
    token: str
    password: str


class OAuthProvider(BaseModel):
    provider: str
    configured: bool
    authorization_url: Optional[str] = None


class OAuthConfigResponse(BaseModel):
    providers: List[OAuthProvider]


class SavedLessonCreate(BaseModel):
    title: str
    result: StudyResponse


class SavedLessonUpdate(BaseModel):
    title: Optional[str] = None
    result: Optional[StudyResponse] = None


class SavedLessonResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    result: StudyResponse


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
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=cookie_secure(),
        samesite=cookie_samesite(),  # type: ignore[arg-type]
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=cookie_secure(),
        samesite=cookie_samesite(),  # type: ignore[arg-type]
    )


def is_expired(expires_at) -> bool:  # type: ignore[no-untyped-def]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=utcnow().tzinfo)
    return expires_at <= utcnow()


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

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


@app.post("/api/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
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
    return AuthResponse(user=user_response(user))


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    email = validate_email(payload.email)
    user = find_user_by_email(db, email)
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(db, user)
    set_session_cookie(response, token)
    return AuthResponse(user=user_response(user))


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> Dict[str, bool]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
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
    db: Session = Depends(get_db),
) -> PasswordResetRequestResponse:
    email = validate_email(payload.email)
    user = find_user_by_email(db, email)
    reset_token = None
    if user is not None:
        token = create_password_reset_token(db, user)
        if os.environ.get("PASSWORD_RESET_RETURN_TOKEN", "").lower() in {"1", "true", "yes"}:
            reset_token = token

    email_configured = bool(os.environ.get("SMTP_HOST") or os.environ.get("RESEND_API_KEY"))
    message = (
        "If an account exists, a reset token was created. Configure SMTP_HOST or RESEND_API_KEY to email it."
        if not email_configured
        else "If an account exists, password reset instructions will be sent."
    )
    return PasswordResetRequestResponse(
        message=message,
        reset_token=reset_token,
        email_delivery_configured=email_configured,
    )


@app.post("/api/auth/password-reset/confirm")
async def confirm_password_reset(payload: PasswordResetConfirmRequest, db: Session = Depends(get_db)) -> Dict[str, bool]:
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
async def oauth_config() -> OAuthConfigResponse:
    google_ready = bool(os.environ.get("GOOGLE_OAUTH_CLIENT_ID") and os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"))
    github_ready = bool(os.environ.get("GITHUB_OAUTH_CLIENT_ID") and os.environ.get("GITHUB_OAUTH_CLIENT_SECRET"))
    return OAuthConfigResponse(
        providers=[
            OAuthProvider(provider="google", configured=google_ready),
            OAuthProvider(provider="github", configured=github_ready),
        ]
    )


@app.get("/api/lessons", response_model=List[SavedLessonResponse])
async def list_lessons(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> List[SavedLessonResponse]:
    lessons = db.scalars(
        select(DBSavedLesson)
        .where(DBSavedLesson.user_id == user.id)
        .order_by(DBSavedLesson.updated_at.desc())
    ).all()
    return [lesson_response(lesson) for lesson in lessons]


@app.post("/api/lessons", response_model=SavedLessonResponse, status_code=status.HTTP_201_CREATED)
async def create_lesson(
    payload: SavedLessonCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonResponse:
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
    lesson = db.get(DBSavedLesson, lesson_id)
    if lesson is None or lesson.user_id != user.id:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    return lesson_response(lesson)


@app.patch("/api/lessons/{lesson_id}", response_model=SavedLessonResponse)
async def update_lesson(
    lesson_id: str,
    payload: SavedLessonUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SavedLessonResponse:
    lesson = db.get(DBSavedLesson, lesson_id)
    if lesson is None or lesson.user_id != user.id:
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
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict[str, bool]:
    lesson = db.get(DBSavedLesson, lesson_id)
    if lesson is None or lesson.user_id != user.id:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    db.delete(lesson)
    db.commit()
    return {"ok": True}


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
