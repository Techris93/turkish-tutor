import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api


class ApiTests(unittest.TestCase):
    def test_study_returns_structured_vocabulary_cards(self):
        client = TestClient(api.app)
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


if __name__ == "__main__":
    unittest.main()

