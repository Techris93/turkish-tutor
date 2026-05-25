"""Structured textbook/PDF lesson breakdown helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List

from content_intelligence import ExtractedContent, TextbookSection, format_textbook_sections_for_prompt


@dataclass(frozen=True)
class TextbookLessonSection:
    title: str
    section_type: str
    source_pages: str
    level: str
    topic: str
    summary: str
    key_vocabulary: List[str]
    grammar_focus: List[str]
    translation: str
    practice: List[str]


def build_textbook_breakdown_json_prompt(
    extracted: ExtractedContent,
    target_language: str,
    cefr_level: str,
) -> str:
    section_context = format_textbook_sections_for_prompt(extracted.textbook_sections, limit=8)
    return f"""You are Turkce Hoca, a Turkish textbook tutor.

The learner uploaded a textbook/PDF. Create a structured lesson breakdown that follows the uploaded content, topics, vocabulary, and grammar signals. The learner's target explanation language is {target_language}. Teach at Turkish CEFR level {cefr_level}, while respecting the source level.

Detected textbook sections:
{section_context}

Return ONLY valid JSON. Do not use Markdown fences.
Use this exact shape:
{{
  "sections": [
    {{
      "title": "source unit or topic title",
      "section_type": "unit/topic|reading|dialogue|grammar|vocabulary|listening|exercise",
      "source_pages": "page range if known",
      "level": "{cefr_level}",
      "topic": "short topic name in {target_language}",
      "summary": "2-3 learner-friendly sentences in {target_language} explaining what this section teaches",
      "key_vocabulary": ["Turkish word or phrase = {target_language} meaning"],
      "grammar_focus": ["grammar point with short {target_language} explanation"],
      "translation": "{target_language} translation or summary of the most important passage from this section",
      "practice": ["short Turkish practice sentence aligned to this section + {target_language} translation"]
    }}
  ]
}}

Rules:
- Include 3 to 8 sections, using the detected source order.
- Do not invent units that are not suggested by the uploaded content.
- Keep examples aligned to the uploaded syllabus/content.
- Preserve Turkish characters.
- If a section is an exercise, explain what skill it practices rather than solving every exercise."""


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end + 1])


def fallback_textbook_breakdown(
    sections: List[TextbookSection],
    target_language: str,
    cefr_level: str,
) -> List[TextbookLessonSection]:
    fallback = []
    for section in sections[:8]:
        terms = [f"{term} = needs {target_language} translation" for term in section.key_terms[:8]]
        grammar = section.grammar_focus[:5] or ["Review sentence endings and suffixes used in the excerpt."]
        excerpt = section.content[:360].strip()
        fallback.append(
            TextbookLessonSection(
                title=section.title,
                section_type=section.section_type,
                source_pages=section.source_pages,
                level=cefr_level or section.level_hint,
                topic=section.title,
                summary=(
                    "This section was detected from the uploaded textbook. Review the vocabulary, grammar signals, "
                    "and excerpt so the lesson stays aligned to the source material."
                ),
                key_vocabulary=terms,
                grammar_focus=grammar,
                translation=f"Needs {target_language} translation. Excerpt: {excerpt}",
                practice=[
                    f"{term}. = Practice this textbook word or phrase." for term in section.key_terms[:3]
                ] or ["Bu bölümden kısa cümleler kur. = Make short sentences from this section."],
            )
        )
    return fallback


def _clean_list(value: Any, fallback: List[str], limit: int = 8) -> List[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item or "").strip()]
        return cleaned[:limit] or fallback
    return fallback


def parse_textbook_breakdown(
    raw_response: str,
    fallback_sections: List[TextbookSection],
    target_language: str,
    cefr_level: str,
) -> tuple[List[TextbookLessonSection], str]:
    fallback = fallback_textbook_breakdown(fallback_sections, target_language, cefr_level)
    try:
        data = _extract_json_object(raw_response)
        raw_sections = data.get("sections", [])
        if not isinstance(raw_sections, list):
            raise ValueError("JSON field 'sections' is not a list.")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return fallback, f"Textbook breakdown JSON could not be parsed; deterministic fallback used. {exc}"

    parsed = []
    for index, source in enumerate(fallback_sections[:8]):
        raw = raw_sections[index] if index < len(raw_sections) and isinstance(raw_sections[index], dict) else {}
        base = fallback[index] if index < len(fallback) else None
        parsed.append(
            TextbookLessonSection(
                title=str(raw.get("title") or source.title).strip()[:140],
                section_type=str(raw.get("section_type") or source.section_type).strip()[:40],
                source_pages=str(raw.get("source_pages") or source.source_pages).strip()[:40],
                level=str(raw.get("level") or cefr_level or source.level_hint).strip()[:8],
                topic=str(raw.get("topic") or source.title).strip()[:140],
                summary=str(raw.get("summary") or (base.summary if base else "")).strip(),
                key_vocabulary=_clean_list(raw.get("key_vocabulary"), base.key_vocabulary if base else [], limit=12),
                grammar_focus=_clean_list(raw.get("grammar_focus"), base.grammar_focus if base else [], limit=8),
                translation=str(raw.get("translation") or (base.translation if base else "")).strip(),
                practice=_clean_list(raw.get("practice"), base.practice if base else [], limit=6),
            )
        )

    warning = ""
    if len(raw_sections) != len(fallback_sections[:8]):
        warning = (
            f"Model returned {len(raw_sections)} textbook sections for {len(fallback_sections[:8])} detected sections; "
            "missing or extra sections were repaired deterministically."
        )
    return parsed, warning
