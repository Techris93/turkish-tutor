import unittest

from content_intelligence import VocabularyItem
from vocabulary_cards import build_translation_lexicon, parse_vocabulary_cards


class VocabularyCardTests(unittest.TestCase):
    def test_parse_repairs_missing_cards_with_fallback(self):
        items = [
            VocabularyItem("arkadaş", "unknown"),
            VocabularyItem("açmak", "verb"),
        ]
        raw = '{"cards": [{"translation": "friend", "example_tr": "Bu benim arkadaşım.", "example_translation": "This is my friend."}]}'
        cards, warning = parse_vocabulary_cards(
            raw,
            items,
            target_language="English",
            cefr_level="A1",
            lexicon={"açmak": "to open"},
        )
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].turkish, "arkadaş")
        self.assertEqual(cards[0].translation, "friend")
        self.assertEqual(cards[1].translation, "to open")
        self.assertIn("repaired", warning)

    def test_parse_invalid_json_uses_fallback_for_all_items(self):
        items = [VocabularyItem("mavi", "adjective/color")]
        cards, warning = parse_vocabulary_cards(
            "not json",
            items,
            target_language="English",
            cefr_level="A1",
            lexicon={"mavi": "blue"},
        )
        self.assertEqual(cards[0].turkish, "mavi")
        self.assertEqual(cards[0].translation, "blue")
        self.assertIn("fallback", warning)

    def test_build_translation_lexicon_reads_curated_lines(self):
        lexicon = build_translation_lexicon([
            {"category": "vocabulary", "content": "Merhaba = Hello\nPardon / Özür dilerim = Excuse me / Sorry"},
            {"category": "grammar", "content": "ignored"},
        ])
        self.assertEqual(lexicon["merhaba"], "Hello")
        self.assertEqual(lexicon["pardon"], "Excuse me / Sorry")
        self.assertEqual(lexicon["özür dilerim"], "Excuse me / Sorry")


if __name__ == "__main__":
    unittest.main()

