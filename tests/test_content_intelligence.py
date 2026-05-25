import tempfile
import unittest
from pathlib import Path

from content_intelligence import (
    build_study_prompt,
    detect_language,
    extract_content,
    extract_textbook_sections,
    extract_turkish_units,
    extract_vocabulary_items,
    infer_cefr_level,
)
from textbook_breakdown import parse_textbook_breakdown


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
            content = extract_content(str(path), current_level="A1", allow_paths=True)
        self.assertEqual(content.source_type, "text-file")
        self.assertEqual(content.inferred_level, "A2")
        self.assertIn("Markete", content.text)

    def test_direct_input_does_not_read_paths_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.txt"
            path.write_text("SECRET_VALUE", encoding="utf-8")
            content = extract_content(str(path), current_level="A1")
        self.assertEqual(content.source_type, "typed-text")
        self.assertNotIn("SECRET_VALUE", content.text)

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

    def test_vocabulary_table_splits_rows_into_items(self):
        text = """İSİMLER FİİLLER
arkadaş çarşı inek mavi salon açmak
anneler günü çay İngiliz mayıs sandalye bakmak
çocuk odası doğum günü cevap vermek tekrar etmek"""
        items = extract_vocabulary_items(text)
        words = [item.text for item in items]
        self.assertNotIn("İSİMLER", words)
        self.assertNotIn("FİİLLER", words)
        self.assertIn("arkadaş", words)
        self.assertIn("çarşı", words)
        self.assertIn("açmak", words)
        self.assertIn("anneler günü", words)
        self.assertIn("çocuk odası", words)
        self.assertIn("doğum günü", words)
        self.assertIn("cevap vermek", words)
        self.assertIn("tekrar etmek", words)

    def test_vocabulary_table_infers_basic_types(self):
        items = {item.text: item.item_type for item in extract_vocabulary_items("mavi Almanya Alman bakmak cevap vermek")}
        self.assertEqual(items["mavi"], "adjective/color")
        self.assertEqual(items["Almanya"], "place/country")
        self.assertEqual(items["Alman"], "nationality")
        self.assertEqual(items["bakmak"], "verb")
        self.assertEqual(items["cevap vermek"], "verb")

    def test_textbook_sections_detect_units_and_focus(self):
        text = """[Page 8]
ÜNİTE 1 Sağlıklı Yaşam
OKUMA
Doktor hastaya düzenli spor yapmasını tavsiye etti. Hasta ilaçlarını kullanmalı.
DİL BİLGİSİ
Gereklilik kipi: -malı / -meli.

[Page 9]
KELİME
sağlık hastane randevu ilaç spor yapmak"""
        sections = extract_textbook_sections(text)
        self.assertGreaterEqual(len(sections), 2)
        self.assertEqual(sections[0].section_type, "reading")
        self.assertIn("Sağlıklı Yaşam", sections[0].title)
        self.assertTrue(any("Gereklilik" in item for item in sections[0].grammar_focus + sections[1].grammar_focus))
        self.assertTrue(any("sağlık" in item.lower() for section in sections for item in section.key_terms))

    def test_textbook_prompt_mentions_source_alignment(self):
        content = extract_content(
            "ÜNİTE 2 Şehir Hayatı\nOKUMA\nOtobüse bindim ve müzeye gittim.",
            current_level="B1",
        )
        prompt = build_study_prompt(content, "English", "B1", "")
        self.assertIn("textbook or syllabus", prompt)
        self.assertIn("Textbook guide", prompt)

    def test_textbook_breakdown_parser_repairs_missing_sections(self):
        sections = extract_textbook_sections(
            "ÜNİTE 1 Ev Hayatı\nKELİME\nmutfak salon kira ödemek komşu taşınmak aidat elektrik faturası"
        )
        parsed, warning = parse_textbook_breakdown('{"sections":[]}', sections, "English", "B1")
        self.assertEqual(len(parsed), len(sections[:8]))
        self.assertIn("Model returned", warning)
        self.assertEqual(parsed[0].level, "B1")


if __name__ == "__main__":
    unittest.main()
