import assert from "node:assert/strict";
import { test } from "node:test";
import {
  SavedLesson,
  StudyResponse,
  VocabularyCard,
  audioCacheKey,
  createSavedLesson,
  deserializeLessons,
  examplePlaybackQueue,
  exampleSegments,
  formatPlaybackRate,
  formatPair,
  normalizePlaybackRate,
  PLAYBACK_RATE_PRESETS,
  playbackProgress,
  readAloudSource,
  serializeLessons,
  shouldUseGeneratedAudio,
  splitSpeechText,
  spokenTextDisplay,
  textQueueItem,
  upsertLesson,
  wordPlaybackQueue,
  wordSegments
} from "./learning";

const card: VocabularyCard = {
  turkish: "gel",
  item_type: "verb",
  translation: "come",
  cefr_level: "A1",
  example_tr: "Buraya gel.",
  example_translation: "Come here.",
  learner_note: "Common command.",
  tts_word: "gel",
  tts_sentence: "Buraya gel."
};

const result: StudyResponse = {
  source_type: "typed-text",
  source_label: "direct input",
  inferred_level: "A1",
  study_level: "A1",
  target_language: "English",
  preview: "gel",
  units: [],
  vocabulary_cards: [card],
  vocabulary_warning: "",
  note: "Study note"
};

test("formatPair formats bilingual read-aloud text", () => {
  assert.equal(formatPair("gel", "come"), "come, gel");
  assert.equal(formatPair("Buraya gel.", "Come here."), "Come here, Buraya gel");
});

test("wordSegments returns translation then Turkish for bilingual mode", () => {
  assert.deepEqual(wordSegments(card, "English", "bilingual"), [
    { text: "come", lang: "en-US" },
    { text: "gel", lang: "tr-TR" }
  ]);
});

test("exampleSegments returns translated example then Turkish example for bilingual mode", () => {
  assert.deepEqual(exampleSegments(card, "English", "bilingual"), [
    { text: "Come here.", lang: "en-US" },
    { text: "Buraya gel.", lang: "tr-TR" }
  ]);
});

test("exampleSegments can return only translation", () => {
  assert.deepEqual(exampleSegments(card, "English", "translation"), [
    { text: "Come here.", lang: "en-US" }
  ]);
});

test("playback queues preserve word and example metadata", () => {
  const words = wordPlaybackQueue([card], "English", "bilingual");
  assert.equal(words.length, 1);
  assert.equal(words[0].title, "gel");
  assert.equal(words[0].subtitle, "come");
  assert.deepEqual(words[0].segments, [
    { text: "come", lang: "en-US" },
    { text: "gel", lang: "tr-TR" }
  ]);

  const examples = examplePlaybackQueue([card], "English", "bilingual");
  assert.equal(examples[0].title, "Buraya gel.");
  assert.equal(examples[0].subtitle, "Come here.");
  assert.deepEqual(examples[0].segments, [
    { text: "Come here.", lang: "en-US" },
    { text: "Buraya gel.", lang: "tr-TR" }
  ]);
});

test("text queue and progress helpers are deterministic", () => {
  assert.deepEqual(textQueueItem("  Merhaba   dünya  "), {
    id: "study-note",
    title: "Study note",
    subtitle: "Türkçe Hoca",
    segments: [{ text: "Merhaba dünya", lang: "tr-TR" }]
  });
  assert.equal(playbackProgress(0, 3), "1 of 3");
  assert.equal(playbackProgress(10, 3), "3 of 3");
  assert.equal(playbackProgress(0, 0), "Ready");
});

test("long read-aloud text is split into smaller speech chunks", () => {
  const longText = "Birinci cümle Türkçe öğrenmek için önemlidir. ".repeat(12);
  const chunks = splitSpeechText(longText, 120);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => chunk.length <= 120));

  const item = textQueueItem(longText);
  assert.ok(item.segments.length > 1);
  assert.ok(item.segments.every((segment) => segment.lang === "tr-TR"));
});

test("readAloudSource explains what the generic read-aloud button will play", () => {
  assert.deepEqual(readAloudSource(null, " Merhaba, bugün Türkçe öğreniyorum. "), {
    label: "Text",
    text: "Merhaba, bugün Türkçe öğreniyorum."
  });

  assert.deepEqual(readAloudSource({ ...result, note: "Summary\n\nListen Practice\n- Gel buraya." }, ""), {
    label: "Practice",
    text: "Gel buraya."
  });

  assert.deepEqual(readAloudSource({ ...result, note: "## Study Note\nTürkçe eklemeli bir dildir." }, ""), {
    label: "Note",
    text: "Study Note Türkçe eklemeli bir dildir."
  });
});

test("spoken text display formats current read-aloud segment", () => {
  const item = examplePlaybackQueue([card], "English", "bilingual")[0];
  assert.deepEqual(spokenTextDisplay(item, item.segments[0], 2, 5, 0), {
    title: "Buraya gel.",
    subtitle: "Come here.",
    text: "Come here.",
    lang: "en-US",
    progress: "3 of 5 · segment 1 of 2"
  });
  assert.equal(spokenTextDisplay(null, item.segments[0], 0, 1), null);
});

test("generated audio cache keys include text, voice, speed, and provider", () => {
  const segment = { text: "gel", lang: "tr-TR" };
  assert.notEqual(audioCacheKey(segment, "openai", "nova", 1), audioCacheKey(segment, "openai", "nova", 1.25));
  assert.notEqual(audioCacheKey(segment, "openai", "nova", 1), audioCacheKey(segment, "openai", "alloy", 1));
  assert.equal(audioCacheKey(segment, "openai", "nova", 1), audioCacheKey(segment, "openai", "nova", 1.004));
});

test("playback rate uses YouTube-style quarter-step presets", () => {
  assert.deepEqual([...PLAYBACK_RATE_PRESETS], [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]);
  assert.equal(normalizePlaybackRate(0.1), 0.25);
  assert.equal(normalizePlaybackRate(0.5), 0.5);
  assert.equal(normalizePlaybackRate(0.63), 0.75);
  assert.equal(normalizePlaybackRate(1.24), 1.25);
  assert.equal(normalizePlaybackRate(2.5), 2);
  assert.equal(normalizePlaybackRate(Number.NaN), 1);
  assert.equal(formatPlaybackRate(0.25), "0.25x");
  assert.equal(formatPlaybackRate(0.5), "0.5x");
  assert.equal(formatPlaybackRate(1.25), "1.25x");
  assert.equal(formatPlaybackRate(1), "1.0x");
  assert.equal(formatPlaybackRate(2), "2.0x");

  const segment = { text: "gel", lang: "tr-TR" };
  assert.equal(audioCacheKey(segment, "openai", "nova", 1.12), audioCacheKey(segment, "openai", "nova", 1));
  assert.notEqual(audioCacheKey(segment, "openai", "nova", 1.13), audioCacheKey(segment, "openai", "nova", 1));
});

test("generated audio engine selection requires explicit generated mode", () => {
  assert.equal(shouldUseGeneratedAudio("generated", true, true), true);
  assert.equal(shouldUseGeneratedAudio("generated", true, false), false);
  assert.equal(shouldUseGeneratedAudio("browser", true, true), false);
  assert.equal(shouldUseGeneratedAudio("browser", false, true), false);
});

test("saved lessons serialize, deserialize, and upsert", () => {
  const lesson = createSavedLesson(result, "Commands");
  const roundTrip = deserializeLessons(serializeLessons([lesson]));
  assert.equal(roundTrip.length, 1);
  assert.equal(roundTrip[0].title, "Commands");
  assert.equal(roundTrip[0].result.vocabulary_cards[0].turkish, "gel");

  const renamed: SavedLesson = { ...lesson, title: "A1 commands" };
  const updated = upsertLesson([lesson], renamed);
  assert.equal(updated.length, 1);
  assert.equal(updated[0].title, "A1 commands");
});

test("deserializeLessons safely ignores invalid data", () => {
  assert.deepEqual(deserializeLessons("not json"), []);
  assert.deepEqual(deserializeLessons(JSON.stringify([{ id: 1 }])), []);
});
