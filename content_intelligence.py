"""
Content extraction and study prompt helpers for Turkce Hoca.

The functions in this module avoid provider lock-in: local file parsing and
heuristics happen here, while Gemini is only used by tutor.py for generation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".srt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
DOC_EXTENSIONS = {".docx"}
TURKISH_CHARACTERS = set("çğıöşüÇĞİÖŞÜ")
COMMON_TURKISH_WORDS = {
    "ben", "sen", "o", "biz", "siz", "onlar", "ve", "bir", "bu", "şu",
    "ile", "için", "de", "da", "mi", "mı", "mu", "mü", "var", "yok",
    "merhaba", "teşekkür", "lütfen", "evet", "hayır", "türkçe",
}


class ExtractionError(RuntimeError):
    """Raised when an input type is recognized but cannot be extracted."""


@dataclass(frozen=True)
class TextUnit:
    text: str
    kind: str
    turkish_signal: bool


@dataclass(frozen=True)
class ExtractedContent:
    source_type: str
    source_label: str
    text: str
    units: List[TextUnit]
    inferred_level: str

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


def _read_pdf(path: Path) -> str:
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
    return "\n\n".join(pages)


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


def extract_text_from_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Extract text from a supported local file and return text plus type."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise ExtractionError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ExtractionError(f"Input is not a file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return _read_text_file(file_path), "text-file"
    if suffix in PDF_EXTENSIONS:
        return _read_pdf(file_path), "pdf"
    if suffix in DOC_EXTENSIONS:
        return _read_docx(file_path), "document"
    if suffix in IMAGE_EXTENSIONS:
        return _read_image(file_path), "image"

    raise ExtractionError(
        f"Unsupported input file type '{suffix}'. Supported: text, PDF, DOCX, images."
    )


def extract_content(raw_input: str, current_level: str = "A1") -> ExtractedContent:
    """Extract content from direct text or a local file path."""
    raw_input = raw_input.strip()
    if not raw_input:
        raise ExtractionError("No input provided.")

    candidate = raw_input[1:] if raw_input.startswith("@") else raw_input
    expanded = Path(candidate).expanduser()
    if expanded.exists():
        text, source_type = extract_text_from_file(expanded)
        label = str(expanded.resolve())
    else:
        text, source_type, label = raw_input, "typed-text", "direct input"

    text = normalize_text(text)
    if not text:
        raise ExtractionError(f"No readable text found in {label}.")

    units = extract_turkish_units(text)
    inferred_level = infer_cefr_level(text, fallback=current_level)
    return ExtractedContent(source_type, label, text, units, inferred_level)


def format_units_for_prompt(units: Iterable[TextUnit], limit: int = 18) -> str:
    lines = []
    for unit in list(units)[:limit]:
        signal = "Turkish signal" if unit.turkish_signal else "general text"
        lines.append(f"- ({unit.kind}, {signal}) {unit.text}")
    return "\n".join(lines) or "- No clean study units detected."


def build_study_prompt(
    extracted: ExtractedContent,
    target_language: str,
    cefr_level: str,
    knowledge_context: str,
) -> str:
    """Build a generation prompt for translation, explanation, and examples."""
    units = format_units_for_prompt(extracted.units)
    return f"""You are Turkce Hoca, a precise Turkish learning assistant.

Analyze the extracted learner input below. The learner's target explanation language is {target_language}.
Teach at CEFR Turkish level {cefr_level}. If the source looks easier or harder, mention that briefly.

Source type: {extracted.source_type}
Inferred source level: {extracted.inferred_level}

Detected study units:
{units}

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
4. {cefr_level} examples: generate 5 new Turkish phrases or sentences using the same vocabulary and grammar, calibrated to {cefr_level}, each with a {target_language} translation.
5. Listen practice: provide 3 short lines that are ideal for text-to-speech practice.

Be direct for translation tasks. Do not hide the answer behind Socratic questions."""

