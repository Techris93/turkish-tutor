import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import api
from auth_storage import configure_database, drop_db, init_db


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
        os.environ["PASSWORD_RESET_RETURN_TOKEN"] = "true"
        self.client = TestClient(api.app)

    def tearDown(self):
        os.environ.pop("PASSWORD_RESET_RETURN_TOKEN", None)
        os.environ.pop("DATABASE_URL", None)
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
        self.assertEqual(login_client.get("/api/auth/me").status_code, 200)

    def test_password_reset_token_updates_password(self):
        self.signup()
        request = self.client.post(
            "/api/auth/password-reset/request",
            json={"email": "learner@example.com"},
        )
        self.assertEqual(request.status_code, 200)
        token = request.json()["reset_token"]
        self.assertTrue(token)

        confirm = self.client.post(
            "/api/auth/password-reset/confirm",
            json={"token": token, "password": "newpassword123"},
        )
        self.assertEqual(confirm.status_code, 200)

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
