import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from auth_storage import configure_database, drop_db, init_db
from rate_limit import limiter
from tts_provider import TTSProviderError, TTSResult


class TTSApiTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self.tmpdir.name) / 'tts.sqlite3'}"
        os.environ["DATABASE_URL"] = self.database_url
        os.environ["RATE_LIMIT_ENABLED"] = "false"
        configure_database(self.database_url)
        drop_db()
        init_db()
        limiter.clear()
        self.client = TestClient(api.app)

    def tearDown(self):
        for key in [
            "DATABASE_URL",
            "RATE_LIMIT_ENABLED",
            "RATE_LIMIT_TTS",
            "TTS_PROVIDER",
            "OPENAI_API_KEY",
            "OPENAI_TTS_MODEL",
            "OPENAI_TTS_VOICE_TR",
            "OPENAI_TTS_VOICE_DEFAULT",
        ]:
            os.environ.pop(key, None)
        limiter.clear()
        self.tmpdir.cleanup()

    def signup(self) -> TestClient:
        client = TestClient(api.app)
        response = client.post(
            "/api/auth/signup",
            json={"email": "tts@example.com", "password": "password123", "name": "TTS User"},
        )
        self.assertEqual(response.status_code, 201)
        return client

    def test_tts_config_reports_missing_provider(self):
        response = self.client.get("/api/tts/config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "none")
        self.assertFalse(payload["configured"])

    def test_tts_audio_requires_authentication(self):
        os.environ["TTS_PROVIDER"] = "mock"
        response = self.client.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(response.status_code, 401)

    def test_tts_audio_missing_provider_config(self):
        signed_in = self.signup()
        response = signed_in.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Generated audio is not configured", response.json()["detail"])

    def test_tts_audio_validation(self):
        signed_in = self.signup()
        os.environ["TTS_PROVIDER"] = "mock"
        empty = signed_in.post("/api/tts/audio", json={"text": "", "language": "tr-TR"})
        self.assertEqual(empty.status_code, 422)
        long_text = "a" * 4097
        too_long = signed_in.post("/api/tts/audio", json={"text": long_text, "language": "tr-TR"})
        self.assertEqual(too_long.status_code, 422)

    def test_tts_audio_mock_provider_returns_audio(self):
        signed_in = self.signup()
        os.environ["TTS_PROVIDER"] = "mock"
        response = signed_in.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-tts-provider"], "mock")
        self.assertGreater(len(response.content), 100)

    def test_tts_audio_maps_provider_failure(self):
        signed_in = self.signup()
        with patch.object(api, "synthesize_tts", new=AsyncMock(side_effect=TTSProviderError("provider down"))):
            response = signed_in.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "provider down")

    def test_tts_audio_mocked_provider_success(self):
        signed_in = self.signup()
        with patch.object(
            api,
            "synthesize_tts",
            new=AsyncMock(return_value=TTSResult(b"audio", "audio/mpeg", "openai", "nova", "gpt-4o-mini-tts")),
        ):
            response = signed_in.post(
                "/api/tts/audio",
                json={"text": "Merhaba", "language": "tr-TR", "provider": "openai"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.headers["x-tts-provider"], "openai")
        self.assertEqual(response.content, b"audio")

    def test_tts_rate_limit_returns_429(self):
        os.environ["RATE_LIMIT_ENABLED"] = "true"
        os.environ["RATE_LIMIT_TTS"] = "1/60s"
        os.environ["TTS_PROVIDER"] = "mock"
        limiter.clear()
        signed_in = self.signup()
        first = signed_in.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(first.status_code, 200)
        second = signed_in.post("/api/tts/audio", json={"text": "Merhaba", "language": "tr-TR"})
        self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
