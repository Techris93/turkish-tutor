import assert from "node:assert/strict";
import test from "node:test";
import {
  PRACTICE_PROGRESS_KEY,
  answerMatches,
  applyPracticeAnswer,
  buildPracticeSession,
  deserializePracticeProgressMap,
  emptyPracticeProgress,
  progressAccuracy,
  serializePracticeProgressMap
} from "./games";
import { StudyResponse } from "./learning";

const study: StudyResponse = {
  source_type: "image",
  source_label: "word-list.jpg",
  inferred_level: "A1",
  study_level: "A1",
  target_language: "English",
  preview: "gel git hata yapmak anneler günü",
  units: [],
  vocabulary_cards: [
    {
      turkish: "gelmek",
      item_type: "verb",
      translation: "to come",
      cefr_level: "A1",
      example_tr: "Buraya gel.",
      example_translation: "Come here.",
      learner_note: "Common movement verb.",
      tts_word: "gelmek",
      tts_sentence: "Buraya gel."
    },
    {
      turkish: "hata yapmak",
      item_type: "phrase",
      translation: "to make a mistake",
      cefr_level: "A2",
      example_tr: "Bazen hata yapmak normal.",
      example_translation: "Sometimes making a mistake is normal.",
      learner_note: "Learn this as a chunk.",
      tts_word: "hata yapmak",
      tts_sentence: "Bazen hata yapmak normal."
    },
    {
      turkish: "anneler günü",
      item_type: "phrase",
      translation: "Mother's Day",
      cefr_level: "A1",
      example_tr: "Anneler günü mayısta.",
      example_translation: "Mother's Day is in May.",
      learner_note: "A useful date phrase.",
      tts_word: "anneler günü",
      tts_sentence: "Anneler günü mayısta."
    },
    {
      turkish: "mavi",
      item_type: "adjective",
      translation: "blue",
      cefr_level: "A1",
      example_tr: "Mavi araba burada.",
      example_translation: "The blue car is here.",
      learner_note: "Color adjectives come before nouns.",
      tts_word: "mavi",
      tts_sentence: "Mavi araba burada."
    }
  ],
  vocabulary_warning: "",
  textbook_sections: [
    {
      title: "Günlük konuşma",
      section_type: "unit",
      source_pages: "1-3",
      level: "A1",
      topic: "Daily phrases",
      summary: "Daily words and chunks.",
      key_vocabulary: ["gelmek", "hata yapmak"],
      grammar_focus: ["chunks"],
      translation: "",
      practice: []
    }
  ],
  note: "Short note."
};

test("buildPracticeSession creates all core activity types from vocabulary cards", () => {
  const session = buildPracticeSession(study, { seed: "fixed", maxQuestions: 30 });
  const activities = new Set(session.questions.map((question) => question.activity));
  assert.equal(session.level, "A1");
  assert.equal(session.topic, "Daily phrases");
  assert.ok(activities.has("match"));
  assert.ok(activities.has("listen"));
  assert.ok(activities.has("recall"));
  assert.ok(activities.has("sentence"));
  assert.ok(activities.has("blank"));
  assert.ok(activities.has("chunk"));
  assert.ok(activities.has("boss"));
});

test("practice questions have useful answers and distractors", () => {
  const session = buildPracticeSession(study, { seed: "choices", maxQuestions: 20 });
  const choiceQuestions = session.questions.filter((question) => ["listen", "recall", "blank", "boss"].includes(question.activity));
  assert.ok(choiceQuestions.length > 0);
  for (const question of choiceQuestions) {
    assert.ok(question.answer);
    assert.ok(question.choices.length >= 2);
    assert.ok(question.choices.some((choice) => choice.text === question.answer));
  }
});

test("sentence builder preserves correct Turkish sentence order", () => {
  const session = buildPracticeSession(study, { mode: "sentence", seed: "sentence", maxQuestions: 5 });
  const question = session.questions.find((item) => item.activity === "sentence" && item.turkish === "mavi");
  assert.ok(question);
  assert.deepEqual(question.answerParts, ["Mavi", "araba", "burada"]);
  assert.ok(answerMatches(question, ["Mavi", "araba", "burada"]));
  assert.equal(answerMatches(question, ["araba", "Mavi", "burada"]), false);
});

test("fill blank questions never remove an empty token", () => {
  const session = buildPracticeSession(study, { mode: "blank", seed: "blank", maxQuestions: 5 });
  for (const question of session.questions) {
    assert.ok(question.blankedText.includes("_____"));
    assert.ok(question.answer.trim().length > 0);
  }
});

test("practice progress updates mastery, missed cards, and accuracy", () => {
  const session = buildPracticeSession(study, { seed: "progress", maxQuestions: 5 });
  const question = session.questions.find((item) => item.activity !== "match");
  assert.ok(question);
  const missed = applyPracticeAnswer(emptyPracticeProgress(), question, false);
  assert.equal(missed.attempts, 1);
  assert.equal(missed.correct, 0);
  assert.equal(missed.masteryByCard[question.cardId], "learning");
  assert.ok(missed.missedCardIds.includes(question.cardId));

  const recovered = applyPracticeAnswer(missed, question, true);
  assert.equal(recovered.correct, 1);
  assert.equal(recovered.masteryByCard[question.cardId], "strong");
  assert.equal(recovered.missedCardIds.includes(question.cardId), false);
  assert.equal(progressAccuracy(recovered), 50);
});

test("local practice progress serialization is resilient", () => {
  const progress = {
    [PRACTICE_PROGRESS_KEY]: emptyPracticeProgress(),
    lesson1: { ...emptyPracticeProgress(), xp: 42, masteryByCard: { "0-gelmek": "mastered" as const } }
  };
  const roundTrip = deserializePracticeProgressMap(serializePracticeProgressMap(progress));
  assert.equal(roundTrip.lesson1.xp, 42);
  assert.equal(roundTrip.lesson1.masteryByCard["0-gelmek"], "mastered");
  assert.deepEqual(deserializePracticeProgressMap("not json"), {});
});
