import tempfile
import unittest
from pathlib import Path

from content_intelligence import (
    detect_language,
    extract_content,
    extract_turkish_units,
    infer_cefr_level,
)


class ContentIntelligenceTests(unittest.TestCase):
    def test_detects_turkish_from_characters(self):
        self.assertEqual(detect_language("Merhaba, Türkçe öğreniyorum."), "tr")

    def test_extracts_direct_input_units(self):
        content = extract_content("Merhaba. Bugün Türkçe öğreniyorum.", current_level="A1")
        self.assertEqual(content.source_type, "typed-text")
        self.assertGreaterEqual(len(content.units), 2)
        self.assertTrue(any(unit.turkish_signal for unit in content.units))

    def test_reads_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesson.txt"
            path.write_text("A2\nMarkete gidiyorum.", encoding="utf-8")
            content = extract_content(str(path), current_level="A1")
        self.assertEqual(content.source_type, "text-file")
        self.assertEqual(content.inferred_level, "A2")
        self.assertIn("Markete", content.text)

    def test_infers_higher_level_for_complex_text(self):
        text = (
            "Özgürlük ve sorumluluk bakımından bu konuyu değerlendirmek gerekir. "
            "Dolayısıyla öğrenciler farklı varsayımları karşılaştırmak zorundadır."
        )
        self.assertIn(infer_cefr_level(text), {"B1", "B2", "C1", "C2"})

    def test_units_fall_back_when_language_is_unknown(self):
        units = extract_turkish_units("Bonjour tout le monde. Hello there.")
        self.assertEqual(len(units), 2)
        self.assertFalse(all(unit.turkish_signal for unit in units))


if __name__ == "__main__":
    unittest.main()

