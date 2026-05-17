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
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth_storage import (
    SESSION_COOKIE_NAME,
    AuthSession,
    OAuthHandoff,
    OAuthState,
    PasswordResetToken,
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
    build_study_prompt,
    extract_content,
    extract_text_from_file,
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
    session_token: Optional[str] = None


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


class OAuthRedeemRequest(BaseModel):
    handoff: str


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


class OAuthRedeemResponse(BaseModel):
    user: UserResponse
    lessons: List[SavedLessonResponse]
    session_token: str


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
    lessons = db.scalars(
        select(DBSavedLesson)
        .where(DBSavedLesson.user_id == user.id)
        .order_by(DBSavedLesson.updated_at.desc())
    ).all()
    return OAuthRedeemResponse(
        user=user_response(user),
        lessons=[lesson_response(lesson) for lesson in lessons],
        session_token=token,
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
    lesson = db.get(DBSavedLesson, lesson_id)
    if lesson is None or lesson.user_id != user.id:
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
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict[str, bool]:
    rate_limit(request, "lesson_write", RateLimitRule(120, 3600), user.id)
    lesson = db.get(DBSavedLesson, lesson_id)
    if lesson is None or lesson.user_id != user.id:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    db.delete(lesson)
    db.commit()
    return {"ok": True}


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
