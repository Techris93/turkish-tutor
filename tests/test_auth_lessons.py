import os
import tempfile
import unittest
from pathlib import Path
from datetime import timedelta
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

import api
from auth_storage import (
    PasswordResetToken,
    configure_database,
    create_oauth_state,
    drop_db,
    hash_token,
    init_db,
    session_factory,
    utcnow,
)
from oauth_flow import OAuthProfile
from rate_limit import limiter


def study_result() -> dict:
    return {
        "source_type": "typed-text",
        "source_label": "direct input",
        "inferred_level": "A1",
        "study_level": "A1",
        "target_language": "English",
        "preview": "gel",
        "units": [{"text": "gel", "kind": "word", "turkish_signal": True}],
        "vocabulary_cards": [
            {
                "turkish": "gel",
                "item_type": "verb",
                "translation": "come",
                "cefr_level": "A1",
                "example_tr": "Buraya gel.",
                "example_translation": "Come here.",
                "learner_note": "Common command.",
                "tts_word": "gel",
                "tts_sentence": "Buraya gel.",
            }
        ],
        "vocabulary_warning": "",
        "note": "Short study note.",
    }


class AuthLessonTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self.tmpdir.name) / 'test.sqlite3'}"
        os.environ["DATABASE_URL"] = self.database_url
        configure_database(self.database_url)
        drop_db()
        init_db()
        limiter.clear()
        os.environ["RATE_LIMIT_ENABLED"] = "false"
        os.environ["PASSWORD_RESET_RETURN_TOKEN"] = "true"
        self.client = TestClient(api.app)

    def tearDown(self):
        for key in [
            "PASSWORD_RESET_RETURN_TOKEN",
            "DATABASE_URL",
            "RATE_LIMIT_ENABLED",
            "RATE_LIMIT_LOGIN",
            "SMTP_HOST",
            "SMTP_FROM_EMAIL",
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GOOGLE_OAUTH_REDIRECT_URI",
            "OAUTH_SUCCESS_REDIRECT_URL",
            "OAUTH_ERROR_REDIRECT_URL",
            "AUTH_COOKIE_SAMESITE",
            "AUTH_COOKIE_SECURE",
            "MAX_UPLOAD_BYTES",
            "MAX_TEXT_INPUT_CHARS",
            "CSRF_PROTECTION_ENABLED",
            "FRONTEND_ORIGIN",
            "RATE_LIMIT_BACKEND",
            "REDIS_URL",
        ]:
            os.environ.pop(key, None)
        limiter.clear()
        self.tmpdir.cleanup()

    def signup(self, email: str = "learner@example.com") -> TestClient:
        client = TestClient(api.app)
        response = client.post(
            "/api/auth/signup",
            json={"email": email, "password": "password123", "name": "Learner"},
        )
        self.assertEqual(response.status_code, 201)
        return client

    def test_signup_duplicate_login_and_current_user(self):
        response = self.client.post(
            "/api/auth/signup",
            json={"email": "learner@example.com", "password": "password123", "name": "Learner"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["user"]["email"], "learner@example.com")
        self.assertTrue(response.json()["session_token"])

        duplicate = self.client.post(
            "/api/auth/signup",
            json={"email": "learner@example.com", "password": "password123", "name": "Learner"},
        )
        self.assertEqual(duplicate.status_code, 409)

        current = self.client.get("/api/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user"]["name"], "Learner")

        bad_login = TestClient(api.app).post(
            "/api/auth/login",
            json={"email": "learner@example.com", "password": "wrongpassword"},
        )
        self.assertEqual(bad_login.status_code, 401)

        login_client = TestClient(api.app)
        login = login_client.post(
            "/api/auth/login",
            json={"email": "learner@example.com", "password": "password123"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json()["session_token"])
        self.assertEqual(login_client.get("/api/auth/me").status_code, 200)

    def test_session_token_header_authenticates_and_logs_out(self):
        signup = self.client.post(
            "/api/auth/signup",
            json={"email": "header@example.com", "password": "password123", "name": "Header User"},
        )
        self.assertEqual(signup.status_code, 201)
        token = signup.json()["session_token"]
        headers = {"X-Session-Token": token}

        header_client = TestClient(api.app)
        current = header_client.get("/api/auth/me", headers=headers)
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user"]["email"], "header@example.com")

        created = header_client.post(
            "/api/lessons",
            headers=headers,
            json={"title": "Header lesson", "result": study_result()},
        )
        self.assertEqual(created.status_code, 201)
        listed = header_client.get("/api/lessons", headers=headers)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)

        logout = header_client.post("/api/auth/logout", headers=headers)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(header_client.get("/api/auth/me", headers=headers).status_code, 401)

    def test_password_reset_token_updates_password(self):
        signed_in = self.signup()
        self.assertEqual(signed_in.get("/api/auth/me").status_code, 200)
        request = signed_in.post(
            "/api/auth/password-reset/request",
            json={"email": "learner@example.com"},
        )
        self.assertEqual(request.status_code, 200)
        token = request.json()["reset_token"]
        self.assertTrue(token)

        confirm = signed_in.post(
            "/api/auth/password-reset/confirm",
            json={"token": token, "password": "newpassword123"},
        )
        self.assertEqual(confirm.status_code, 200)
        self.assertEqual(signed_in.get("/api/auth/me").status_code, 401)

        old_login = TestClient(api.app).post(
            "/api/auth/login",
            json={"email": "learner@example.com", "password": "password123"},
        )
        self.assertEqual(old_login.status_code, 401)

        new_client = TestClient(api.app)
        new_login = new_client.post(
            "/api/auth/login",
            json={"email": "learner@example.com", "password": "newpassword123"},
        )
        self.assertEqual(new_login.status_code, 200)

    def test_password_reset_email_send_path_and_missing_provider(self):
        self.signup()
        os.environ["PASSWORD_RESET_RETURN_TOKEN"] = "false"
        os.environ.pop("SMTP_HOST", None)
        missing = self.client.post(
            "/api/auth/password-reset/request",
            json={"email": "learner@example.com"},
        )
        self.assertEqual(missing.status_code, 200)
        self.assertFalse(missing.json()["email_delivery_configured"])
        self.assertIsNone(missing.json()["reset_token"])

        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_FROM_EMAIL"] = "noreply@example.com"
        with patch.object(api, "send_password_reset_email") as send_mock:
            response = self.client.post(
                "/api/auth/password-reset/request",
                json={"email": "learner@example.com"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["email_delivery_configured"])
        send_mock.assert_called_once()

    def test_password_reset_token_is_single_use_and_expires(self):
        self.signup()
        request = self.client.post(
            "/api/auth/password-reset/request",
            json={"email": "learner@example.com"},
        )
        token = request.json()["reset_token"]
        first = self.client.post(
            "/api/auth/password-reset/confirm",
            json={"token": token, "password": "newpassword123"},
        )
        self.assertEqual(first.status_code, 200)
        second = self.client.post(
            "/api/auth/password-reset/confirm",
            json={"token": token, "password": "anotherpassword123"},
        )
        self.assertEqual(second.status_code, 400)

        request = self.client.post(
            "/api/auth/password-reset/request",
            json={"email": "learner@example.com"},
        )
        expired_token = request.json()["reset_token"]
        with session_factory()() as db:
            reset = db.get(PasswordResetToken, hash_token(expired_token))
            reset.expires_at = utcnow() - timedelta(minutes=1)
            db.commit()
        expired = self.client.post(
            "/api/auth/password-reset/confirm",
            json={"token": expired_token, "password": "anotherpassword123"},
        )
        self.assertEqual(expired.status_code, 400)

    def test_oauth_config_start_invalid_state_and_callback_login(self):
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "google-client"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "google-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "http://testserver/api/auth/oauth/google/callback"
        os.environ["OAUTH_SUCCESS_REDIRECT_URL"] = "http://localhost:3000/?oauth=success"

        config = self.client.get("/api/auth/oauth/config")
        self.assertEqual(config.status_code, 200)
        google = config.json()["providers"][0]
        self.assertTrue(google["configured"])
        self.assertTrue(google["authorization_url"].endswith("/api/auth/oauth/google/start"))

        start = self.client.get("/api/auth/oauth/google/start", follow_redirects=False)
        self.assertEqual(start.status_code, 302)
        location = start.headers["location"]
        self.assertIn("accounts.google.com", location)
        state = parse_qs(urlparse(location).query)["state"][0]

        invalid = self.client.get(
            "/api/auth/oauth/google/callback?code=abc&state=bad",
            follow_redirects=False,
        )
        self.assertEqual(invalid.status_code, 400)

        with patch.object(
            api,
            "exchange_oauth_profile",
            new=AsyncMock(return_value=OAuthProfile("oauth@example.com", "OAuth Learner", "provider-id")),
        ):
            callback = self.client.get(
                f"/api/auth/oauth/google/callback?code=abc&state={state}",
                follow_redirects=False,
            )
        self.assertEqual(callback.status_code, 302)
        callback_url = urlparse(callback.headers["location"])
        callback_params = parse_qs(callback_url.query)
        self.assertEqual(callback_params["oauth"], ["success"])
        handoff = callback_params["handoff"][0]
        current = self.client.get("/api/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user"]["email"], "oauth@example.com")

        redeem_client = TestClient(api.app)
        redeemed = redeem_client.post("/api/auth/oauth/redeem", json={"handoff": handoff})
        self.assertEqual(redeemed.status_code, 200)
        self.assertEqual(redeemed.json()["user"]["email"], "oauth@example.com")
        session_token = redeemed.json()["session_token"]
        self.assertTrue(session_token)
        self.assertEqual(redeem_client.get("/api/auth/me").status_code, 200)

        header_client = TestClient(api.app)
        headers = {"X-Session-Token": session_token}
        self.assertEqual(header_client.get("/api/auth/me", headers=headers).status_code, 200)
        saved = header_client.post(
            "/api/lessons",
            headers=headers,
            json={"title": "OAuth saved lesson", "result": study_result()},
        )
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(len(header_client.get("/api/lessons", headers=headers).json()), 1)

        reused_handoff = TestClient(api.app).post("/api/auth/oauth/redeem", json={"handoff": handoff})
        self.assertEqual(reused_handoff.status_code, 400)

        reused = self.client.get(
            f"/api/auth/oauth/google/callback?code=abc&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(reused.status_code, 400)

    def test_oauth_callback_rejects_expired_state(self):
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "google-client"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "google-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "http://testserver/api/auth/oauth/google/callback"
        with session_factory()() as db:
            state = create_oauth_state(db, "google")
            saved = db.get(api.OAuthState, hash_token(state))
            saved.expires_at = utcnow() - timedelta(minutes=1)
            db.commit()
        expired = self.client.get(
            f"/api/auth/oauth/google/callback?code=abc&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(expired.status_code, 400)

    def test_oauth_redeem_rejects_invalid_and_expired_handoff(self):
        self.assertEqual(
            self.client.post("/api/auth/oauth/redeem", json={"handoff": "missing-handoff-token"}).status_code,
            400,
        )
        signed_in = self.signup("handoff@example.com")
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "google-client"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "google-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "http://testserver/api/auth/oauth/google/callback"
        start = signed_in.get("/api/auth/oauth/google/start", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        with patch.object(
            api,
            "exchange_oauth_profile",
            new=AsyncMock(return_value=OAuthProfile("handoff@example.com", "Handoff", "provider-id")),
        ):
            callback = signed_in.get(
                f"/api/auth/oauth/google/callback?code=abc&state={state}",
                follow_redirects=False,
            )
        handoff = parse_qs(urlparse(callback.headers["location"]).query)["handoff"][0]
        with session_factory()() as db:
            saved = db.get(api.OAuthHandoff, hash_token(handoff))
            saved.expires_at = utcnow() - timedelta(minutes=1)
            db.commit()
        expired = TestClient(api.app).post("/api/auth/oauth/redeem", json={"handoff": handoff})
        self.assertEqual(expired.status_code, 400)

    def test_oauth_does_not_switch_signed_in_account_to_different_email(self):
        signed_in = self.signup("password@example.com")
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "google-client"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "google-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "http://testserver/api/auth/oauth/google/callback"
        os.environ["OAUTH_ERROR_REDIRECT_URL"] = "http://localhost:3000/?oauth=error"

        start = signed_in.get("/api/auth/oauth/google/start", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        with patch.object(
            api,
            "exchange_oauth_profile",
            new=AsyncMock(return_value=OAuthProfile("google@example.com", "Google User", "provider-id")),
        ):
            callback = signed_in.get(
                f"/api/auth/oauth/google/callback?code=abc&state={state}",
                follow_redirects=False,
            )
        self.assertEqual(callback.status_code, 302)
        self.assertIn("reason=account_mismatch", callback.headers["location"])
        current = signed_in.get("/api/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user"]["email"], "password@example.com")

    def test_login_rate_limit_returns_429(self):
        os.environ["RATE_LIMIT_ENABLED"] = "true"
        os.environ["RATE_LIMIT_LOGIN"] = "2/60s"
        limiter.clear()
        for _ in range(2):
            response = self.client.post(
                "/api/auth/login",
                json={"email": "missing@example.com", "password": "password123"},
            )
            self.assertEqual(response.status_code, 401)
        limited = self.client.post(
            "/api/auth/login",
            json={"email": "missing@example.com", "password": "password123"},
        )
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Too many requests", limited.json()["detail"])

    def test_security_headers_and_samesite_none_secure_cookie(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.headers["x-content-type-options"], "nosniff")
        self.assertEqual(health.headers["x-frame-options"], "DENY")
        self.assertIn("camera=()", health.headers["permissions-policy"])

        os.environ["AUTH_COOKIE_SAMESITE"] = "none"
        os.environ["AUTH_COOKIE_SECURE"] = "false"
        response = TestClient(api.app).post(
            "/api/auth/signup",
            json={"email": "cookie@example.com", "password": "password123", "name": "Cookie User"},
        )
        self.assertEqual(response.status_code, 201)
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("samesite=none", cookie)
        self.assertIn("secure", cookie)

    def test_csrf_origin_guard_blocks_cookie_mutation_without_trusted_origin(self):
        os.environ["CSRF_PROTECTION_ENABLED"] = "true"
        os.environ["FRONTEND_ORIGIN"] = "http://localhost:3000"
        owner = self.signup("csrf@example.com")

        blocked = owner.post("/api/lessons", json={"title": "Blocked", "result": study_result()})
        self.assertEqual(blocked.status_code, 403)
        self.assertIn("Cross-site request rejected", blocked.json()["detail"])

        allowed = owner.post(
            "/api/lessons",
            headers={"Origin": "http://localhost:3000"},
            json={"title": "Allowed", "result": study_result()},
        )
        self.assertEqual(allowed.status_code, 201)

    def test_redis_rate_limit_backend_requires_url(self):
        os.environ["RATE_LIMIT_ENABLED"] = "true"
        os.environ["RATE_LIMIT_BACKEND"] = "redis"
        response = self.client.post(
            "/api/auth/login",
            json={"email": "missing@example.com", "password": "password123"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("REDIS_URL is missing", response.json()["detail"])

    def test_study_rejects_unsupported_and_oversized_uploads(self):
        unsupported = self.client.post(
            "/api/study",
            data={"level": "A1", "target_language": "English"},
            files={"file": ("payload.exe", b"not text", "application/octet-stream")},
        )
        self.assertEqual(unsupported.status_code, 400)
        self.assertIn("Unsupported input file type", unsupported.json()["detail"])

        os.environ["MAX_UPLOAD_BYTES"] = "4"
        too_large = self.client.post(
            "/api/study",
            data={"level": "A1", "target_language": "English"},
            files={"file": ("lesson.txt", b"Merhaba", "text/plain")},
        )
        self.assertEqual(too_large.status_code, 400)
        self.assertIn("too large", too_large.json()["detail"])

    def test_study_rejects_oversized_text_input(self):
        os.environ["MAX_TEXT_INPUT_CHARS"] = "5"
        response = self.client.post(
            "/api/study",
            data={"text": "Merhaba", "level": "A1", "target_language": "English"},
        )
        self.assertEqual(response.status_code, 413)

    def test_study_text_flow_does_not_resolve_server_paths(self):
        secret_path = Path(self.tmpdir.name) / "secret.txt"
        secret_path.write_text("SECRET_VALUE", encoding="utf-8")
        with (
            patch.object(api, "extract_vocabulary_items", return_value=[]),
            patch.object(api, "ask_llm", new=AsyncMock(return_value="Safe study note.")),
        ):
            response = self.client.post(
                "/api/study",
                data={"text": str(secret_path), "level": "A1", "target_language": "English"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_type"], "typed-text")
        self.assertIn("secret.txt", payload["preview"])
        self.assertNotIn("SECRET_VALUE", payload["preview"])

    def test_saved_lesson_crud_and_cross_user_protection(self):
        owner = self.signup("owner@example.com")
        create = owner.post(
            "/api/lessons",
            json={"title": "A1 commands", "result": study_result()},
        )
        self.assertEqual(create.status_code, 201)
        lesson = create.json()
        self.assertEqual(lesson["result"]["vocabulary_cards"][0]["turkish"], "gel")

        listed = owner.get("/api/lessons")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)

        updated = owner.patch(f"/api/lessons/{lesson['id']}", json={"title": "A1 verbs"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["title"], "A1 verbs")

        other = self.signup("other@example.com")
        self.assertEqual(other.get(f"/api/lessons/{lesson['id']}").status_code, 404)
        self.assertEqual(other.patch(f"/api/lessons/{lesson['id']}", json={"title": "Mine"}).status_code, 404)
        self.assertEqual(other.delete(f"/api/lessons/{lesson['id']}").status_code, 404)

        deleted = owner.delete(f"/api/lessons/{lesson['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(owner.get("/api/lessons").json(), [])


if __name__ == "__main__":
    unittest.main()
