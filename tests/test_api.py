import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from auth_storage import configure_database, drop_db, init_db


class ApiTests(unittest.TestCase):
    def test_study_returns_structured_vocabulary_cards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATABASE_URL"] = f"sqlite:///{Path(tmpdir) / 'api.sqlite3'}"
            try:
                configure_database(os.environ["DATABASE_URL"])
                drop_db()
                init_db()
                client = TestClient(api.app)
                self._assert_study_cards(client)
                self._assert_textbook_sections(client)
            finally:
                os.environ.pop("DATABASE_URL", None)

    def test_study_uses_fallback_cards_when_provider_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATABASE_URL"] = f"sqlite:///{Path(tmpdir) / 'api.sqlite3'}"
            try:
                configure_database(os.environ["DATABASE_URL"])
                drop_db()
                init_db()
                client = TestClient(api.app)
                with patch.object(
                    api,
                    "ask_llm",
                    new=AsyncMock(side_effect=RuntimeError("Gemini request failed: 503 UNAVAILABLE")),
                ):
                    response = client.post(
                        "/api/study",
                        data={
                            "text": "arkadaş açmak mavi",
                            "level": "A1",
                            "target_language": "English",
                        },
                    )
            finally:
                os.environ.pop("DATABASE_URL", None)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["vocabulary_cards"]), 3)
        self.assertEqual(payload["vocabulary_cards"][2]["translation"], "blue")
        self.assertIn("deterministic fallback", payload["vocabulary_warning"])
        self.assertNotIn("{'error'", payload["vocabulary_warning"])
        self.assertIn("deterministic fallback", payload["note"])

    def _assert_study_cards(self, client: TestClient):
        card_json = """
        {
          "cards": [
            {
              "turkish": "arkadaş",
              "item_type": "noun",
              "translation": "friend",
              "cefr_level": "A1",
              "example_tr": "Bu benim arkadaşım.",
              "example_translation": "This is my friend.",
              "learner_note": "A common people word.",
              "tts_word": "arkadaş",
              "tts_sentence": "Bu benim arkadaşım."
            },
            {
              "turkish": "açmak",
              "item_type": "verb",
              "translation": "to open",
              "cefr_level": "A1",
              "example_tr": "Kapıyı aç.",
              "example_translation": "Open the door.",
              "learner_note": "Infinitive ending is -mak.",
              "tts_word": "açmak",
              "tts_sentence": "Kapıyı aç."
            }
          ]
        }
        """
        with patch.object(api, "ask_llm", new=AsyncMock(side_effect=[card_json, "Short study summary."])):
            response = client.post(
                "/api/study",
                data={
                    "text": "İSİMLER FİİLLER\narkadaş açmak",
                    "level": "A1",
                    "target_language": "English",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["note"], "Short study summary.")
        self.assertEqual(len(payload["vocabulary_cards"]), 2)
        self.assertEqual(payload["vocabulary_cards"][0]["turkish"], "arkadaş")
        self.assertEqual(payload["vocabulary_cards"][1]["example_tr"], "Kapıyı aç.")

    def _assert_textbook_sections(self, client: TestClient):
        textbook_json = """
        {
          "sections": [
            {
              "title": "ÜNİTE 1 Sağlıklı Yaşam",
              "section_type": "unit/topic",
              "source_pages": "p. 8",
              "level": "B1",
              "topic": "Healthy living",
              "summary": "This unit teaches health vocabulary and advice forms.",
              "key_vocabulary": ["sağlık = health", "randevu = appointment"],
              "grammar_focus": ["-malı/-meli = necessity"],
              "translation": "A short passage about healthy routines.",
              "practice": ["Düzenli spor yapmalıyım. = I should exercise regularly."]
            }
          ]
        }
        """
        with (
            patch.object(api, "extract_vocabulary_items", return_value=[]),
            patch.object(api, "ask_llm", new=AsyncMock(side_effect=[textbook_json, "Textbook study note."])),
        ):
            response = client.post(
                "/api/study",
                data={
                    "text": "ÜNİTE 1 Sağlıklı Yaşam\nOKUMA\nDoktor hastaya düzenli spor yapmasını tavsiye etti. Hasta ilaçlarını kullanmalı.",
                    "level": "B1",
                    "target_language": "English",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["note"], "Textbook study note.")
        self.assertEqual(payload["textbook_sections"][0]["topic"], "Healthy living")
        self.assertEqual(payload["textbook_sections"][0]["grammar_focus"][0], "-malı/-meli = necessity")


if __name__ == "__main__":
    unittest.main()
