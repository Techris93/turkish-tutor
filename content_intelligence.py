"""
Content extraction and study prompt helpers for Turkce Hoca.

The functions in this module avoid provider lock-in: local file parsing and
heuristics happen here, while Gemini is only used by tutor.py for generation.
"""

from __future__ import annotations

import os
import re
import unicodedata
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".srt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
DOC_EXTENSIONS = {".docx"}
SUPPORTED_FILE_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS | DOC_EXTENSIONS
TURKISH_CHARACTERS = set("çğıöşüÇĞİÖŞÜ")
COMMON_TURKISH_WORDS = {
    "ben", "sen", "o", "biz", "siz", "onlar", "ve", "bir", "bu", "şu",
    "ile", "için", "de", "da", "mi", "mı", "mu", "mü", "var", "yok",
    "merhaba", "teşekkür", "lütfen", "evet", "hayır", "türkçe",
}
VOCABULARY_HEADINGS = {
    "isim", "isimler", "fiil", "fiiller", "sifat", "sıfat", "sıfatlar",
    "renk", "renkler", "kelime", "kelimeler", "vocabulary", "nouns",
    "verbs", "adjectives", "tr", "turkce", "türkçe",
}
KNOWN_MULTIWORD_ITEMS = {
    "anneler günü",
    "çocuk odası",
    "doğum günü",
    "cevap vermek",
    "tekrar etmek",
    "seyahat etmek",
    "rica etmek",
    "telefon etmek",
    "yardım etmek",
    "alışveriş yapmak",
    "hoşça kal",
    "güle güle",
    "iyi günler",
    "iyi akşamlar",
    "iyi geceler",
}
COMPOUND_VERB_AUXILIARIES = {"etmek", "vermek", "olmak", "yapmak", "kalmak"}
SAFE_OCR_CORRECTIONS = {
    "aksam": "akşam",
    "ingiliz": "İngiliz",
    "ingiltere": "İngiltere",
    "iran": "İran",
    "italya": "İtalya",
    "nigeria": "Nijerya",
    "ginli": "Çinli",
    "gin": "Çin",
}
COLORS = {
    "beyaz", "siyah", "mavi", "kırmızı", "sarı", "yeşil", "turuncu",
    "mor", "pembe", "kahverengi", "gri",
}
COUNTRIES_AND_PLACES = {
    "afrika", "asya", "avrupa", "amerika", "almanya", "arnavutluk",
    "çin", "ingiltere", "iran", "ispanya", "mısır", "nijerya",
    "somali", "suriye", "japonya", "italya",
}
NATIONALITIES = {
    "alman", "arap", "arnavut", "ingiliz", "iranlı", "ispanyol",
    "mısırlı", "çinli", "japon", "somalili", "suriyeli",
}


class ExtractionError(RuntimeError):
    """Raised when an input type is recognized but cannot be extracted."""


@dataclass(frozen=True)
class TextUnit:
    text: str
    kind: str
    turkish_signal: bool


@dataclass(frozen=True)
class VocabularyItem:
    text: str
    item_type: str


@dataclass(frozen=True)
class TextbookSection:
    title: str
    section_type: str
    level_hint: str
    source_pages: str
    content: str
    key_terms: List[str]
    grammar_focus: List[str]


@dataclass(frozen=True)
class ExtractedContent:
    source_type: str
    source_label: str
    text: str
    units: List[TextUnit]
    inferred_level: str
    textbook_sections: List[TextbookSection]
    extraction_warning: str = ""

    @property
    def preview(self) -> str:
        text = self.text.strip()
        return text[:700] + ("..." if len(text) > 700 else "")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving sentence punctuation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_vocab_item(text: str) -> str:
    """Clean a single OCR vocabulary item without over-correcting content."""
    text = re.sub(r"[^\wÇĞİÖŞÜçğıöşü' -]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    if not text:
        return ""

    words = []
    for word in text.split():
        lower = word.lower()
        words.append(SAFE_OCR_CORRECTIONS.get(lower, word))
    return " ".join(words)


def vocabulary_key(text: str) -> str:
    """Casefold key that handles Turkish dotted capital I from OCR headings."""
    key = unicodedata.normalize("NFKD", text.casefold())
    key = "".join(ch for ch in key if not unicodedata.combining(ch))
    return key


def _tokenize_vocab_line(line: str) -> List[str]:
    line = re.sub(r"[|•·,;:()\[\]{}]", " ", line)
    raw_tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü']+", line)
    tokens = []
    for token in raw_tokens:
        cleaned = normalize_vocab_item(token)
        if not cleaned:
            continue
        if vocabulary_key(cleaned) in VOCABULARY_HEADINGS:
            continue
        tokens.append(cleaned)
    return tokens


def infer_vocab_type(item: str) -> str:
    """Infer a lightweight vocabulary category for display/filtering."""
    lower = vocabulary_key(item)
    words = lower.split()
    color_keys = {vocabulary_key(value) for value in COLORS}
    place_keys = {vocabulary_key(value) for value in COUNTRIES_AND_PLACES}
    nationality_keys = {vocabulary_key(value) for value in NATIONALITIES}
    multiword_keys = {vocabulary_key(value) for value in KNOWN_MULTIWORD_ITEMS}
    if lower in color_keys:
        return "adjective/color"
    if lower in place_keys:
        return "place/country"
    if lower in nationality_keys:
        return "nationality"
    if lower in multiword_keys:
        if words[-1:] and words[-1] in COMPOUND_VERB_AUXILIARIES:
            return "verb"
        return "phrase"
    if words and words[-1] in COMPOUND_VERB_AUXILIARIES:
        return "verb"
    if lower.endswith(("mak", "mek")):
        return "verb"
    if len(words) > 1:
        return "phrase"
    return "unknown"


def extract_vocabulary_items(text: str, max_items: int = 240) -> List[VocabularyItem]:
    """Extract individual vocabulary entries from OCR tables and word lists.

    This intentionally differs from sentence segmentation: it treats lines like
    "arkadaş çarşı inek mavi salon açmak" as separate vocabulary cells while
    preserving known compounds such as "doğum günü" and "cevap vermek".
    """
    clean = normalize_text(text)
    if not clean:
        return []

    items: List[VocabularyItem] = []
    seen = set()
    multiword_keys = {vocabulary_key(value) for value in KNOWN_MULTIWORD_ITEMS}

    for line in clean.splitlines():
        tokens = _tokenize_vocab_line(line)
        index = 0
        while index < len(tokens):
            chosen = ""
            for size in (3, 2):
                phrase = " ".join(tokens[index:index + size])
                if vocabulary_key(phrase) in multiword_keys:
                    chosen = phrase
                    index += size
                    break

            if not chosen:
                if (
                    index + 1 < len(tokens)
                    and vocabulary_key(tokens[index + 1]) in COMPOUND_VERB_AUXILIARIES
                    and not vocabulary_key(tokens[index]).endswith(("mak", "mek"))
                ):
                    chosen = f"{tokens[index]} {tokens[index + 1]}"
                    index += 2
                else:
                    chosen = tokens[index]
                    index += 1

            chosen = normalize_vocab_item(chosen)
            key = vocabulary_key(chosen)
            if not chosen or key in VOCABULARY_HEADINGS or key in seen:
                continue
            seen.add(key)
            items.append(VocabularyItem(chosen, infer_vocab_type(chosen)))
            if len(items) >= max_items:
                return items

    if len(items) <= 1:
        for segment in segment_text(clean, max_units=max_items):
            item = normalize_vocab_item(segment)
            if item and len(item.split()) <= 4 and item.casefold() not in seen:
                seen.add(item.casefold())
                items.append(VocabularyItem(item, infer_vocab_type(item)))

    return items[:max_items]


def detect_language(text: str) -> str:
    """Lightweight language detector for routing TTS and study defaults."""
    lowered = text.lower()
    if any(ch in text for ch in TURKISH_CHARACTERS):
        return "tr"
    tokens = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+", lowered))
    if len(tokens & COMMON_TURKISH_WORDS) >= 2:
        return "tr"
    if re.search(r"\b(hello|the|and|you|that|this|with|from)\b", lowered):
        return "en"
    return "auto"


def _has_turkish_signal(text: str) -> bool:
    lowered = text.lower()
    if any(ch in text for ch in TURKISH_CHARACTERS):
        return True
    tokens = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+", lowered))
    if tokens & COMMON_TURKISH_WORDS:
        return True
    return bool(re.search(r"\b\w+(lar|ler|dir|dır|dur|dür|iyor|acak|ecek)\b", lowered))


def segment_text(text: str, max_units: int = 40) -> List[str]:
    """Split text into sentence-like study units without heavy NLP deps."""
    clean = normalize_text(text)
    if not clean:
        return []

    raw_segments = re.split(r"(?<=[.!?…])\s+|\n+", clean)
    segments: List[str] = []
    for segment in raw_segments:
        segment = segment.strip(" -\t")
        if not segment:
            continue
        if len(segment) > 420:
            pieces = re.split(r"(?<=[,;:])\s+", segment)
            segments.extend(piece.strip() for piece in pieces if piece.strip())
        else:
            segments.append(segment)
    return segments[:max_units]


def extract_turkish_units(text: str, max_units: int = 24) -> List[TextUnit]:
    """Detect Turkish-looking words, phrases, and sentences from extracted text."""
    segments = segment_text(text, max_units=80)
    if not segments:
        return []

    units: List[TextUnit] = []
    for segment in segments:
        words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü']+", segment)
        if not words:
            continue
        if len(words) == 1:
            kind = "word"
        elif len(words) <= 6 and not re.search(r"[.!?…]$", segment):
            kind = "phrase"
        else:
            kind = "sentence"
        units.append(TextUnit(segment, kind, _has_turkish_signal(segment)))

    turkish_units = [unit for unit in units if unit.turkish_signal]
    return (turkish_units or units)[:max_units]


def infer_cefr_level(text: str, fallback: str = "A1") -> str:
    """Infer a likely CEFR level from explicit labels and rough complexity."""
    explicit = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", text.upper())
    if explicit:
        return explicit.group(1)

    clean = normalize_text(text)
    if not clean:
        return fallback if fallback in CEFR_LEVELS else "A1"

    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü']+", clean)
    sentences = max(1, len(segment_text(clean, max_units=200)))
    avg_sentence_len = len(words) / sentences
    long_words = sum(1 for word in words if len(word) >= 11)
    advanced_markers = len(re.findall(
        r"\b(rağmen|oysa|halbuki|dolayısıyla|nitekim|itibaren|bakımından|"
        r"değerlendirmek|karşılaştırmak|varsayım|sorumluluk|özgürlük)\b",
        clean.lower(),
    ))

    score = 0
    if len(words) > 70:
        score += 1
    if avg_sentence_len > 8:
        score += 1
    if avg_sentence_len > 14:
        score += 1
    if long_words >= 4:
        score += 1
    if advanced_markers >= 2:
        score += 1

    if score <= 0:
        return "A1"
    if score == 1:
        return "A2"
    if score == 2:
        return "B1"
    if score == 3:
        return "B2"
    if score == 4:
        return "C1"
    return "C2"


def _read_text_file(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1254", "latin-1")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError(f"Could not decode text file: {path}")


def _pdf_ocr_page_limit() -> int:
    raw = os.getenv("PDF_OCR_MAX_PAGES", "16")
    try:
        return max(1, min(int(raw), 60))
    except ValueError:
        return 16


def _read_scanned_pdf(path: Path) -> tuple[str, str]:
    try:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise ExtractionError(
            "This PDF appears to be scanned or image-based. Scanned PDF OCR requires PyMuPDF, Pillow, "
            "pytesseract, and the Tesseract OCR app."
        ) from exc

    try:
        document = fitz.open(str(path))
    except Exception as exc:  # pragma: no cover - library-specific error subclasses vary
        raise ExtractionError(f"Could not open scanned PDF: {path}") from exc

    page_count = len(document)
    max_pages = min(page_count, _pdf_ocr_page_limit())
    pages = []
    try:
        for index in range(max_pages):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image, lang="tur+eng")
            if text.strip():
                pages.append(f"[Page {index + 1}]\n{text}")
    except (OSError, RuntimeError) as exc:
        raise ExtractionError(f"OCR failed for scanned PDF: {path}") from exc
    finally:
        document.close()

    warning = ""
    if page_count > max_pages:
        warning = (
            f"Scanned PDF OCR processed the first {max_pages} of {page_count} pages. "
            "Increase PDF_OCR_MAX_PAGES to process more pages, but expect slower analysis."
        )
    return "\n\n".join(pages), warning


def _read_pdf(path: Path) -> tuple[str, str]:
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
    except ImportError as exc:
        raise ExtractionError(
            "PDF extraction requires pypdf. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index}]\n{text}")
    text = "\n\n".join(pages)
    if len(text.strip()) >= 200:
        return text, ""

    ocr_text, warning = _read_scanned_pdf(path)
    if ocr_text.strip():
        return ocr_text, warning or "PDF had little embedded text, so scanned-page OCR was used."
    return text, "PDF had little embedded text and OCR did not find readable text."


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ExtractionError(
            "DOCX extraction requires python-docx. Install with: pip install python-docx"
        ) from exc

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_image(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise ExtractionError(
            "Image OCR requires Pillow, pytesseract, and the Tesseract OCR app. "
            "Install Python packages with: pip install Pillow pytesseract"
        ) from exc

    try:
        image = Image.open(path)
        return pytesseract.image_to_string(image, lang="tur+eng")
    except (OSError, RuntimeError) as exc:
        raise ExtractionError(f"OCR failed for image: {path}") from exc


def extract_text_from_file_details(path: str | os.PathLike[str]) -> tuple[str, str, str]:
    """Extract text from a supported local file and return text, type, and warning."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise ExtractionError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ExtractionError(f"Input is not a file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return _read_text_file(file_path), "text-file", ""
    if suffix in PDF_EXTENSIONS:
        text, warning = _read_pdf(file_path)
        return text, "pdf", warning
    if suffix in DOC_EXTENSIONS:
        return _read_docx(file_path), "document", ""
    if suffix in IMAGE_EXTENSIONS:
        return _read_image(file_path), "image", ""

    raise ExtractionError(
        f"Unsupported input file type '{suffix}'. Supported: text, PDF, DOCX, images."
    )


def extract_text_from_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Extract text from a supported local file and return text plus type."""
    text, source_type, _warning = extract_text_from_file_details(path)
    return text, source_type


def _page_range_for_text(text: str) -> str:
    pages = re.findall(r"\[Page\s+(\d+)\]", text)
    if not pages:
        return ""
    unique = []
    for page in pages:
        if page not in unique:
            unique.append(page)
    if len(unique) == 1:
        return f"p. {unique[0]}"
    return f"pp. {unique[0]}-{unique[-1]}"


def _section_type_for_title(title: str, body: str) -> str:
    haystack = f"{title}\n{body}".lower()
    haystack_key = vocabulary_key(haystack)
    if re.search(r"\b(dil bilgisi|gramer|grammar|ekler|zaman|kip|fiilimsi)\b", haystack_key):
        return "grammar"
    if re.search(r"\b(kelime|sozluk|vocabulary|deyim|ifadeler)\b", haystack_key):
        return "vocabulary"
    if re.search(r"\b(dinleme|listen|ses)\b", haystack_key):
        return "listening"
    if re.search(r"\b(okuma|metin|reading)\b", haystack_key):
        return "reading"
    if re.search(r"\b(konusma|dialog|diyalog|speaking)\b", haystack_key):
        return "dialogue"
    if re.search(r"\b(alistirma|etkinlik|exercise|soru)\b", haystack_key):
        return "exercise"
    return "unit/topic"


def _guess_grammar_focus(text: str, max_items: int = 5) -> List[str]:
    patterns = [
        (r"\b\w+(?:iyor|ıyor|uyor|üyor)\b", "Şimdiki zaman (-iyor)"),
        (r"\b\w+(?:dı|di|du|dü|tı|ti|tu|tü)\b", "Geçmiş zaman (-di)"),
        (r"\b\w+(?:acak|ecek)\b", "Gelecek zaman (-acak/-ecek)"),
        (r"\b\w+(?:malı|meli)\b", "Gereklilik kipi (-malı/-meli)"),
        (r"\b\w+(?:dan|den|tan|ten)\b", "Ayrılma hali (-dan/-den)"),
        (r"\b\w+(?:da|de|ta|te)\b", "Bulunma hali (-da/-de)"),
        (r"\b\w+(?:a|e)\b", "Yönelme hali (-a/-e)"),
        (r"\b\w+(?:ın|in|un|ün|nın|nin|nun|nün)\b", "İyelik/tamlayan ekleri"),
        (r"\b\w+(?:en|an|dığı|diği|duğu|düğü)\b", "Sıfat-fiil ve isim-fiil yapıları"),
    ]
    found = []
    for pattern, label in patterns:
        if re.search(pattern, text.lower()) and label not in found:
            found.append(label)
        if len(found) >= max_items:
            break
    return found


def extract_textbook_sections(text: str, max_sections: int = 10) -> List[TextbookSection]:
    """Segment textbook-like extracted text into syllabus-oriented sections."""
    clean = normalize_text(text)
    if not clean:
        return []

    heading_pattern = re.compile(
        r"(?im)^(?:\[Page\s+\d+\]\s*)?(?P<title>"
        r"(?:\d+\.\s*)?(?:ÜNİTE|UNIT|BÖLÜM|DERS)\s*\d*[:.\-\s]*[^\n]{0,80}|"
        r"(?:OKUMA|DİNLEME|KONUŞMA|YAZMA|DİL BİLGİSİ|KELİME|SÖZLÜK|GRAMER|ALIŞTIRMA)[^\n]{0,80})$"
    )
    matches = list(heading_pattern.finditer(clean))
    chunks: List[tuple[str, str]] = []
    if matches:
        current_unit = ""
        for index, match in enumerate(matches[: max_sections + 1]):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
            title = normalize_text(match.group("title"))
            body = normalize_text(clean[start:end])
            if re.match(r"(?i)^(?:\d+\.\s*)?(?:ÜNİTE|UNIT|BÖLÜM|DERS)\b", title):
                current_unit = title
                if len(body) < 120:
                    continue
            display_title = f"{current_unit} · {title}" if current_unit and title != current_unit else title
            if len(body) >= 30:
                chunks.append((display_title, body))
            if len(chunks) >= max_sections:
                break

    if not chunks:
        page_chunks = re.split(r"(?=\[Page\s+\d+\])", clean)
        for index, chunk in enumerate(page_chunks):
            chunk = normalize_text(chunk)
            if len(chunk) < 60:
                continue
            title_match = re.search(r"(?m)^(?!\[Page\b)([A-ZÇĞİÖŞÜ0-9][^\n]{4,80})$", chunk)
            title = normalize_text(title_match.group(1)) if title_match else f"Textbook section {len(chunks) + 1}"
            chunks.append((title, chunk))
            if len(chunks) >= max_sections:
                break

    sections = []
    for title, body in chunks:
        vocabulary = [item.text for item in extract_vocabulary_items(body, max_items=12)]
        sections.append(
            TextbookSection(
                title=title[:120],
                section_type=_section_type_for_title(title, body),
                level_hint=infer_cefr_level(f"{title}\n{body}", fallback="B1"),
                source_pages=_page_range_for_text(body),
                content=body[:1600],
                key_terms=vocabulary,
                grammar_focus=_guess_grammar_focus(body),
            )
        )
    return sections


def extract_content(raw_input: str, current_level: str = "A1", allow_paths: bool = False) -> ExtractedContent:
    """Extract content from direct text or a local file path."""
    raw_input = raw_input.strip()
    if not raw_input:
        raise ExtractionError("No input provided.")

    candidate = raw_input[1:] if raw_input.startswith("@") else raw_input
    expanded = Path(candidate).expanduser()
    if allow_paths and expanded.exists():
        text, source_type, extraction_warning = extract_text_from_file_details(expanded)
        label = str(expanded.resolve())
    else:
        text, source_type, label, extraction_warning = raw_input, "typed-text", "direct input", ""

    text = normalize_text(text)
    if not text:
        raise ExtractionError(f"No readable text found in {label}.")

    units = extract_turkish_units(text)
    inferred_level = infer_cefr_level(text, fallback=current_level)
    textbook_sections = extract_textbook_sections(text)
    return ExtractedContent(source_type, label, text, units, inferred_level, textbook_sections, extraction_warning)


def format_units_for_prompt(units: Iterable[TextUnit], limit: int = 18) -> str:
    lines = []
    for unit in list(units)[:limit]:
        signal = "Turkish signal" if unit.turkish_signal else "general text"
        lines.append(f"- ({unit.kind}, {signal}) {unit.text}")
    return "\n".join(lines) or "- No clean study units detected."


def format_textbook_sections_for_prompt(sections: Iterable[TextbookSection], limit: int = 6) -> str:
    lines = []
    for section in list(sections)[:limit]:
        terms = ", ".join(section.key_terms[:8]) or "not detected"
        grammar = ", ".join(section.grammar_focus[:5]) or "not detected"
        lines.append(
            f"- {section.title} ({section.section_type}, {section.level_hint}, {section.source_pages or 'pages unknown'})\n"
            f"  Key terms: {terms}\n"
            f"  Grammar signals: {grammar}\n"
            f"  Excerpt: {section.content[:700]}"
        )
    return "\n".join(lines) or "- No textbook sections detected."


def build_study_prompt(
    extracted: ExtractedContent,
    target_language: str,
    cefr_level: str,
    knowledge_context: str,
) -> str:
    """Build a generation prompt for translation, explanation, and examples."""
    units = format_units_for_prompt(extracted.units)
    textbook_sections = format_textbook_sections_for_prompt(extracted.textbook_sections)
    textbook_instruction = ""
    if extracted.textbook_sections:
        textbook_instruction = f"""
This source looks like a textbook or syllabus. Teach from the detected textbook sections below and keep examples aligned to those topics, vocabulary, and grammar signals.

Detected textbook sections:
{textbook_sections}
"""
    return f"""You are Turkce Hoca, a precise Turkish learning assistant.

Analyze the extracted learner input below. The learner's target explanation language is {target_language}.
Teach at CEFR Turkish level {cefr_level}. If the source looks easier or harder, mention that briefly.

Source type: {extracted.source_type}
Inferred source level: {extracted.inferred_level}
Extraction note: {extracted.extraction_warning or "none"}

Detected study units:
{units}
{textbook_instruction}

Extracted text preview:
\"\"\"
{extracted.preview}
\"\"\"

Relevant tutor knowledge:
{knowledge_context or "(No local knowledge context matched; rely on general Turkish expertise.)"}

Return a polished study note with these sections:
1. Extracted Turkish: list the most useful words, phrases, or sentences.
2. Translation: translate each item into {target_language}.
3. Learner explanation: explain vocabulary, grammar, suffixes, and meaning in a friendly way.
4. Textbook guide: if this is a textbook/PDF, explain the detected unit/topic, grammar focus, and how the learner should study this material.
5. {cefr_level} examples: generate 5 new Turkish phrases or sentences using the same vocabulary and grammar, calibrated to {cefr_level}, each with a {target_language} translation.
6. Listen practice: provide 3 short lines that are ideal for text-to-speech practice.

Be direct for translation tasks. Do not hide the answer behind Socratic questions."""
