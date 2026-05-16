import unittest

from speech import _parse_say_voice_line, normalize_language


class SpeechTests(unittest.TestCase):
    def test_normalizes_language_aliases(self):
        self.assertEqual(normalize_language("Turkish"), "tr")
        self.assertEqual(normalize_language("en-US"), "en")

    def test_auto_language_uses_text(self):
        self.assertEqual(normalize_language("auto", "Bugün hava güzel."), "tr")

    def test_parses_macos_voice_line(self):
        voice = _parse_say_voice_line("Yelda               tr_TR    # Merhaba, benim adım Yelda.")
        self.assertIsNotNone(voice)
        self.assertEqual(voice.name, "Yelda")
        self.assertEqual(voice.language, "tr-tr")


if __name__ == "__main__":
    unittest.main()

