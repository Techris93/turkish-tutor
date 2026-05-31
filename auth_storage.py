from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from sqlalchemy import DateTime, ForeignKey, Index, String, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_URL = f"sqlite:///{REPO_DIR / 'data' / 'turkish_tutor.sqlite3'}"
SESSION_COOKIE_NAME = "turkce_hoca_session"

password_hasher = PasswordHasher()
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None
_configured_url = ""


class Base(DeclarativeBase):
    pass


class JSONVariant(JSON):
    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: utcnow(),
        onupdate=lambda: utcnow(),
        nullable=False,
    )

    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    lessons: Mapped[list["SavedLesson"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (
        Index("ix_auth_sessions_expires_at", "expires_at"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped[User] = relationship(back_populates="sessions")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    __table_args__ = (
        Index("ix_password_reset_tokens_expires_at", "expires_at"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship()


class OAuthState(Base):
    __tablename__ = "oauth_states"
    __table_args__ = (
        Index("ix_oauth_states_provider_expires", "provider", "expires_at"),
    )

    state_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OAuthHandoff(Base):
    __tablename__ = "oauth_handoffs"
    __table_args__ = (
        Index("ix_oauth_handoffs_expires_at", "expires_at"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship()


class SavedLesson(Base):
    __tablename__ = "saved_lessons"
    __table_args__ = (
        Index("ix_saved_lessons_user_updated", "user_id", "updated_at"),
        Index("ix_saved_lessons_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    result: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSONVariant), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utcnow(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: utcnow(),
        onupdate=lambda: utcnow(),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="lessons")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def database_url() -> str:
    return normalize_database_url(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def configure_database(url: Optional[str] = None) -> None:
    global _configured_url, _engine, _session_factory

    resolved = normalize_database_url(url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    if _engine is not None and resolved == _configured_url:
        return
    if _engine is not None:
        _engine.dispose()

    connect_args = {}
    engine_kwargs = {"pool_pre_ping": True}
    if resolved.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if resolved == "sqlite:///:memory:":
            engine_kwargs["poolclass"] = StaticPool
        else:
            db_path = resolved.removeprefix("sqlite:///")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        engine_kwargs.update(
            {
                "pool_size": _int_env("DB_POOL_SIZE", 5, minimum=1),
                "max_overflow": _int_env("DB_MAX_OVERFLOW", 5, minimum=0),
                "pool_timeout": _int_env("DB_POOL_TIMEOUT", 30, minimum=1),
                "pool_recycle": _int_env("DB_POOL_RECYCLE_SECONDS", 1800, minimum=60),
            }
        )

    _engine = create_engine(resolved, connect_args=connect_args, **engine_kwargs)
    _session_factory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    _configured_url = resolved


def engine() -> Engine:
    configure_database()
    assert _engine is not None
    return _engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine())
    ensure_runtime_indexes()


def ensure_runtime_indexes() -> None:
    bind = engine()
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            index.create(bind=bind, checkfirst=True)


def drop_db() -> None:
    Base.metadata.drop_all(bind=engine())


def session_factory() -> sessionmaker[Session]:
    configure_database()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def db_session() -> Iterator[Session]:
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    with db_session() as session:
        yield session


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError):
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    expires_days = int(os.environ.get("AUTH_SESSION_DAYS", "30"))
    db.add(
        AuthSession(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=utcnow() + timedelta(days=expires_days),
        )
    )
    db.commit()
    return token


def create_password_reset_token(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    expires_minutes = int(os.environ.get("PASSWORD_RESET_MINUTES", "30"))
    db.add(
        PasswordResetToken(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=utcnow() + timedelta(minutes=expires_minutes),
        )
    )
    db.commit()
    return token


def create_oauth_state(db: Session, provider: str) -> str:
    state = secrets.token_urlsafe(32)
    expires_minutes = int(os.environ.get("OAUTH_STATE_MINUTES", "10"))
    db.add(
        OAuthState(
            state_hash=hash_token(state),
            provider=provider,
            expires_at=utcnow() + timedelta(minutes=expires_minutes),
        )
    )
    db.commit()
    return state


def create_oauth_handoff(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    expires_minutes = int(os.environ.get("OAUTH_HANDOFF_MINUTES", "5"))
    db.add(
        OAuthHandoff(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=utcnow() + timedelta(minutes=expires_minutes),
        )
    )
    db.commit()
    return token


def find_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
