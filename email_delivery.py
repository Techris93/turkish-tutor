from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from urllib.parse import urlencode


class EmailDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    from_email: str
    use_tls: bool


def smtp_configured() -> bool:
    required = ["SMTP_HOST", "SMTP_FROM_EMAIL"]
    return all(os.environ.get(key) for key in required)


def email_delivery_configured() -> bool:
    return smtp_configured()


def smtp_config() -> EmailConfig:
    host = os.environ.get("SMTP_HOST", "").strip()
    from_email = os.environ.get("SMTP_FROM_EMAIL", "").strip()
    if not host or not from_email:
        raise EmailDeliveryError("Password reset email is not configured.")

    return EmailConfig(
        host=host,
        port=int(os.environ.get("SMTP_PORT", "587")),
        username=os.environ.get("SMTP_USERNAME", "").strip(),
        password=os.environ.get("SMTP_PASSWORD", ""),
        from_email=from_email,
        use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"},
    )


def password_reset_base_url() -> str:
    return (
        os.environ.get("PASSWORD_RESET_BASE_URL")
        or os.environ.get("FRONTEND_ORIGIN")
        or "http://localhost:3000"
    ).strip().rstrip("/")


def build_password_reset_link(token: str) -> str:
    return f"{password_reset_base_url()}/?{urlencode({'reset_token': token})}"


def send_password_reset_email(to_email: str, token: str) -> None:
    config = smtp_config()
    reset_link = build_password_reset_link(token)
    message = EmailMessage()
    message["Subject"] = "Reset your Turkce Hoca password"
    message["From"] = config.from_email
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                "You requested a password reset for Turkce Hoca.",
                "",
                f"Open this link to choose a new password: {reset_link}",
                "",
                "This link expires soon. If you did not request it, you can ignore this email.",
            ]
        )
    )

    try:
        with smtplib.SMTP(config.host, config.port, timeout=10) as smtp:
            if config.use_tls:
                smtp.starttls()
            if config.username or config.password:
                smtp.login(config.username, config.password)
            smtp.send_message(message)
    except OSError as exc:
        raise EmailDeliveryError("Could not send password reset email.") from exc
