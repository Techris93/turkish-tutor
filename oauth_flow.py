from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx


class OAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthProfile:
    email: str
    name: str
    provider_id: str


@dataclass(frozen=True)
class OAuthProviderConfig:
    provider: str
    client_id: str
    client_secret: str
    redirect_uri: str
    auth_url: str
    token_url: str
    scopes: str


def frontend_origin() -> str:
    return (os.environ.get("FRONTEND_ORIGIN") or "http://localhost:3000").strip().rstrip("/")


def oauth_success_redirect_url() -> str:
    return (os.environ.get("OAUTH_SUCCESS_REDIRECT_URL") or f"{frontend_origin()}/?oauth=success").strip()


def oauth_error_redirect_url(reason: str = "oauth_failed") -> str:
    base = (os.environ.get("OAUTH_ERROR_REDIRECT_URL") or f"{frontend_origin()}/?oauth=error").strip()
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{urlencode({'reason': reason})}"


def provider_config(provider: str) -> Optional[OAuthProviderConfig]:
    provider = provider.lower()
    if provider == "google":
        client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
        redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", "").strip()
        if not (client_id and client_secret and redirect_uri):
            return None
        return OAuthProviderConfig(
            provider="google",
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            scopes="openid email profile",
        )
    if provider == "github":
        client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "")
        redirect_uri = os.environ.get("GITHUB_OAUTH_REDIRECT_URI", "").strip()
        if not (client_id and client_secret and redirect_uri):
            return None
        return OAuthProviderConfig(
            provider="github",
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            scopes="read:user user:email",
        )
    return None


def configured_providers() -> Dict[str, bool]:
    return {
        "google": provider_config("google") is not None,
        "github": provider_config("github") is not None,
    }


def authorization_url(provider: str, state: str) -> str:
    config = provider_config(provider)
    if config is None:
        raise OAuthError("OAuth provider is not configured.")
    params = {
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "response_type": "code",
        "scope": config.scopes,
        "state": state,
    }
    if config.provider == "google":
        params["access_type"] = "offline"
        params["prompt"] = "select_account"
        params["nonce"] = secrets.token_urlsafe(16)
    return f"{config.auth_url}?{urlencode(params)}"


async def exchange_oauth_profile(provider: str, code: str) -> OAuthProfile:
    config = provider_config(provider)
    if config is None:
        raise OAuthError("OAuth provider is not configured.")

    async with httpx.AsyncClient(timeout=12) as client:
        token_response = await client.post(
            config.token_url,
            data={
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "redirect_uri": config.redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        if token_response.status_code >= 400:
            raise OAuthError("OAuth token exchange failed.")
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise OAuthError("OAuth provider did not return an access token.")

        if config.provider == "google":
            return await fetch_google_profile(client, access_token)
        if config.provider == "github":
            return await fetch_github_profile(client, access_token)
    raise OAuthError("Unsupported OAuth provider.")


async def fetch_google_profile(client: httpx.AsyncClient, access_token: str) -> OAuthProfile:
    response = await client.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise OAuthError("Could not fetch Google profile.")
    payload = response.json()
    email = str(payload.get("email") or "").strip().lower()
    if not email:
        raise OAuthError("Google account did not provide an email address.")
    return OAuthProfile(
        email=email,
        name=str(payload.get("name") or email.split("@", 1)[0]),
        provider_id=str(payload.get("sub") or email),
    )


async def fetch_github_profile(client: httpx.AsyncClient, access_token: str) -> OAuthProfile:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
    user_response = await client.get("https://api.github.com/user", headers=headers)
    if user_response.status_code >= 400:
        raise OAuthError("Could not fetch GitHub profile.")
    user_payload: Dict[str, Any] = user_response.json()
    email = str(user_payload.get("email") or "").strip().lower()
    if not email:
        email_response = await client.get("https://api.github.com/user/emails", headers=headers)
        if email_response.status_code >= 400:
            raise OAuthError("Could not fetch GitHub email address.")
        for item in email_response.json():
            if item.get("primary") and item.get("verified") and item.get("email"):
                email = str(item["email"]).strip().lower()
                break
    if not email:
        raise OAuthError("GitHub account did not provide a verified email address.")
    return OAuthProfile(
        email=email,
        name=str(user_payload.get("name") or user_payload.get("login") or email.split("@", 1)[0]),
        provider_id=str(user_payload.get("id") or email),
    )
