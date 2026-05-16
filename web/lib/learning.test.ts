import assert from "node:assert/strict";
import { test } from "node:test";
import {
  SavedLesson,
  StudyResponse,
  VocabularyCard,
  createSavedLesson,
  deserializeLessons,
  exampleSegments,
  formatPair,
  serializeLessons,
  upsertLesson,
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
  assert.equal(formatPair("gel", "come"), "gel, come");
  assert.equal(formatPair("Buraya gel.", "Come here."), "Buraya gel, Come here.");
});

test("wordSegments returns Turkish then translation for bilingual mode", () => {
  assert.deepEqual(wordSegments(card, "English", "bilingual"), [
    { text: "gel", lang: "tr-TR" },
    { text: "come", lang: "en-US" }
  ]);
});

test("exampleSegments can return only translation", () => {
  assert.deepEqual(exampleSegments(card, "English", "translation"), [
    { text: "Come here.", lang: "en-US" }
  ]);
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
