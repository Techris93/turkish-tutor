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
  note: string;
};

export type SavedLesson = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  result: StudyResponse;
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

export const SAVED_LESSONS_KEY = "turkce-hoca.saved-lessons.v1";

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

export function formatPair(primary: string, translation: string): string {
  const left = normalizeSpeechText(primary).replace(/[.!?]+$/, "");
  const right = normalizeSpeechText(translation);
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
    { text: normalizeSpeechText(card.tts_word || card.turkish), lang: "tr-TR" },
    { text: normalizeSpeechText(card.translation), lang: languageCode(targetLanguage) }
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
    { text: normalizeSpeechText(card.tts_sentence || card.example_tr), lang: "tr-TR" },
    { text: normalizeSpeechText(card.example_translation), lang: languageCode(targetLanguage) }
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
  return queueItem("study-note", title, "Turkce Hoca", [{ text: normalizeSpeechText(text), lang: "tr-TR" }]);
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

export function audioCacheKey(segment: SpeechSegment, provider: string, voice: string, speed: number): string {
  return JSON.stringify({
    text: segment.text,
    lang: segment.lang,
    provider: provider || "auto",
    voice: voice || "auto",
    speed: Number(speed.toFixed(2))
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
