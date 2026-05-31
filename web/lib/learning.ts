export type PlaybackMode = "turkish" | "translation" | "bilingual";
export type PlaybackEngine = "generated" | "browser";

export type StudyUnit = {
  text: string;
  kind: string;
  turkish_signal: boolean;
};

export type VocabularyCard = {
  turkish: string;
  item_type: string;
  translation: string;
  cefr_level: string;
  example_tr: string;
  example_translation: string;
  learner_note: string;
  tts_word: string;
  tts_sentence: string;
};

export type TextbookSection = {
  title: string;
  section_type: string;
  source_pages: string;
  level: string;
  topic: string;
  summary: string;
  key_vocabulary: string[];
  grammar_focus: string[];
  translation: string;
  practice: string[];
};

export type StudyResponse = {
  source_type: string;
  source_label: string;
  inferred_level: string;
  study_level: string;
  target_language: string;
  preview: string;
  units: StudyUnit[];
  vocabulary_cards: VocabularyCard[];
  vocabulary_warning: string;
  textbook_sections?: TextbookSection[];
  textbook_warning?: string;
  extraction_warning?: string;
  note: string;
};

export type SavedLesson = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  result?: StudyResponse;
};

export type SavedLessonListResponse = {
  lessons: SavedLesson[];
  limit: number;
  offset: number;
  total: number;
};

export type SpeechSegment = {
  text: string;
  lang: string;
};

export type PlaybackQueueItem = {
  id: string;
  title: string;
  subtitle: string;
  segments: SpeechSegment[];
};

export type SpokenTextDisplay = {
  title: string;
  subtitle: string;
  text: string;
  lang: string;
  progress: string;
};

export const SAVED_LESSONS_KEY = "turkce-hoca.saved-lessons.v1";
export const MIN_PLAYBACK_RATE = 0.25;
export const MAX_PLAYBACK_RATE = 2;
export const PLAYBACK_RATE_STEP = 0.25;
export const PLAYBACK_RATE_PRESETS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] as const;
export const MAX_SPEECH_SEGMENT_CHARS = 220;

const targetLanguageCodes: Record<string, string> = {
  english: "en-US",
  turkish: "tr-TR",
  spanish: "es-ES",
  french: "fr-FR",
  german: "de-DE",
  italian: "it-IT"
};

export function languageCode(language: string): string {
  return targetLanguageCodes[language.trim().toLowerCase()] ?? "en-US";
}

function normalizeSpeechText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function splitSpeechText(text: string, maxChars = MAX_SPEECH_SEGMENT_CHARS): string[] {
  const clean = normalizeSpeechText(text);
  if (!clean) {
    return [];
  }
  if (clean.length <= maxChars) {
    return [clean];
  }

  const sentences = clean.match(/[^.!?…]+[.!?…]?/g)?.map((item) => normalizeSpeechText(item)).filter(Boolean) ?? [clean];
  const chunks: string[] = [];
  let current = "";

  for (const sentence of sentences) {
    if (!current) {
      current = sentence;
      continue;
    }
    if (`${current} ${sentence}`.length <= maxChars) {
      current = `${current} ${sentence}`;
    } else {
      chunks.push(current);
      current = sentence;
    }
  }
  if (current) {
    chunks.push(current);
  }

  return chunks.flatMap((chunk) => {
    if (chunk.length <= maxChars) {
      return [chunk];
    }
    const pieces: string[] = [];
    let current = "";
    for (const word of chunk.split(/\s+/)) {
      if (!current) {
        current = word;
      } else if (`${current} ${word}`.length <= maxChars) {
        current = `${current} ${word}`;
      } else {
        pieces.push(current);
        current = word;
      }
    }
    if (current) {
      pieces.push(current);
    }
    return pieces.filter(Boolean);
  });
}

export function formatPair(primary: string, translation: string): string {
  const left = normalizeSpeechText(translation).replace(/[.!?]+$/, "");
  const right = normalizeSpeechText(primary).replace(/[.!?]+$/, "");
  if (left && right) {
    return `${left}, ${right}`;
  }
  return left || right;
}

export function wordSegments(card: VocabularyCard, targetLanguage: string, mode: PlaybackMode): SpeechSegment[] {
  if (mode === "turkish") {
    return [{ text: normalizeSpeechText(card.tts_word || card.turkish), lang: "tr-TR" }];
  }
  if (mode === "translation") {
    return [{ text: normalizeSpeechText(card.translation), lang: languageCode(targetLanguage) }];
  }
  return [
    { text: normalizeSpeechText(card.translation), lang: languageCode(targetLanguage) },
    { text: normalizeSpeechText(card.tts_word || card.turkish), lang: "tr-TR" }
  ].filter((segment) => segment.text);
}

export function exampleSegments(card: VocabularyCard, targetLanguage: string, mode: PlaybackMode): SpeechSegment[] {
  if (mode === "turkish") {
    return [{ text: normalizeSpeechText(card.tts_sentence || card.example_tr), lang: "tr-TR" }];
  }
  if (mode === "translation") {
    return [{ text: normalizeSpeechText(card.example_translation), lang: languageCode(targetLanguage) }];
  }
  return [
    { text: normalizeSpeechText(card.example_translation), lang: languageCode(targetLanguage) },
    { text: normalizeSpeechText(card.tts_sentence || card.example_tr), lang: "tr-TR" }
  ].filter((segment) => segment.text);
}

function queueItem(id: string, title: string, subtitle: string, segments: SpeechSegment[]): PlaybackQueueItem {
  return {
    id,
    title: normalizeSpeechText(title),
    subtitle: normalizeSpeechText(subtitle),
    segments: segments.filter((segment) => normalizeSpeechText(segment.text)).map((segment) => ({
      ...segment,
      text: normalizeSpeechText(segment.text)
    }))
  };
}

export function textQueueItem(text: string, title = "Study note"): PlaybackQueueItem {
  return queueItem(
    "study-note",
    title,
    "Türkçe Hoca",
    splitSpeechText(text).map((chunk) => ({ text: chunk, lang: "tr-TR" }))
  );
}

export function readAloudSource(result: StudyResponse | null | undefined, inputText: string): { label: string; text: string } {
  if (!result) {
    return {
      label: "Text",
      text: normalizeSpeechText(inputText)
    };
  }

  const listenPractice = result.note.split(/listen practice/i)[1];
  return {
    label: listenPractice ? "Practice" : "Note",
    text: normalizeSpeechText((listenPractice || result.note).replace(/[#*_`>-]/g, " "))
  };
}

export function wordPlaybackQueue(
  cards: VocabularyCard[],
  targetLanguage: string,
  mode: PlaybackMode
): PlaybackQueueItem[] {
  return cards.map((card, index) =>
    queueItem(
      `word-${index}-${card.turkish}`,
      card.turkish,
      card.translation || card.item_type,
      wordSegments(card, targetLanguage, mode)
    )
  );
}

export function examplePlaybackQueue(
  cards: VocabularyCard[],
  targetLanguage: string,
  mode: PlaybackMode
): PlaybackQueueItem[] {
  return cards.map((card, index) =>
    queueItem(
      `example-${index}-${card.turkish}`,
      card.example_tr || card.turkish,
      card.example_translation || card.translation,
      exampleSegments(card, targetLanguage, mode)
    )
  );
}

export function playbackProgress(currentIndex: number, total: number): string {
  if (total <= 0) {
    return "Ready";
  }
  const current = Math.min(Math.max(currentIndex + 1, 1), total);
  return `${current} of ${total}`;
}

export function spokenTextDisplay(
  item: PlaybackQueueItem | null,
  segment: SpeechSegment | null,
  itemIndex: number,
  totalItems: number,
  segmentIndex = 0
): SpokenTextDisplay | null {
  if (!item || !segment?.text) {
    return null;
  }
  const segmentTotal = Math.max(item.segments.length, 1);
  const segmentProgress = segmentTotal > 1 ? ` · segment ${Math.min(segmentIndex + 1, segmentTotal)} of ${segmentTotal}` : "";
  return {
    title: item.title || "Read Aloud",
    subtitle: item.subtitle,
    text: normalizeSpeechText(segment.text),
    lang: segment.lang,
    progress: `${playbackProgress(itemIndex, totalItems)}${segmentProgress}`
  };
}

export function normalizePlaybackRate(rate: number): number {
  if (!Number.isFinite(rate)) {
    return 1;
  }
  const bounded = Math.min(Math.max(rate, MIN_PLAYBACK_RATE), MAX_PLAYBACK_RATE);
  const stepped = Math.round(bounded / PLAYBACK_RATE_STEP) * PLAYBACK_RATE_STEP;
  return Number(stepped.toFixed(2));
}

export function formatPlaybackRate(rate: number): string {
  const normalized = normalizePlaybackRate(rate);
  if (Number.isInteger(normalized)) {
    return `${normalized.toFixed(1)}x`;
  }
  if (Number.isInteger(normalized * 2)) {
    return `${normalized.toFixed(1)}x`;
  }
  return `${normalized.toFixed(2)}x`;
}

export function audioCacheKey(segment: SpeechSegment, provider: string, voice: string, speed: number): string {
  return JSON.stringify({
    text: segment.text,
    lang: segment.lang,
    provider: provider || "auto",
    voice: voice || "auto",
    speed: normalizePlaybackRate(speed)
  });
}

export function shouldUseGeneratedAudio(
  engine: PlaybackEngine,
  generatedConfigured: boolean,
  signedIn: boolean
): boolean {
  return engine === "generated" && generatedConfigured && signedIn;
}

export function createSavedLesson(result: StudyResponse, title?: string): SavedLesson {
  const now = new Date().toISOString();
  const fallbackTitle = [
    result.source_label && result.source_label !== "direct input" ? result.source_label : "Turkish lesson",
    result.study_level
  ]
    .filter(Boolean)
    .join(" · ");

  return {
    id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    title: (title || fallbackTitle).trim(),
    created_at: now,
    updated_at: now,
    result
  };
}

export function serializeLessons(lessons: SavedLesson[]): string {
  return JSON.stringify(lessons);
}

export function deserializeLessons(raw: string | null): SavedLesson[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((lesson): lesson is SavedLesson => {
      return (
        lesson &&
        typeof lesson.id === "string" &&
        typeof lesson.title === "string" &&
        typeof lesson.created_at === "string" &&
        typeof lesson.updated_at === "string" &&
        lesson.result &&
        Array.isArray(lesson.result.vocabulary_cards)
      );
    });
  } catch {
    return [];
  }
}

export function upsertLesson(lessons: SavedLesson[], lesson: SavedLesson): SavedLesson[] {
  const existingIndex = lessons.findIndex((item) => item.id === lesson.id);
  if (existingIndex === -1) {
    return [lesson, ...lessons];
  }
  const next = [...lessons];
  next[existingIndex] = { ...lesson, updated_at: new Date().toISOString() };
  return next;
}
