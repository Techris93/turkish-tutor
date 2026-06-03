import assert from "node:assert/strict";
import test from "node:test";
import {
  PRACTICE_PROGRESS_KEY,
  answerMatches,
  applyPracticeAnswer,
  buildPracticeSession,
  deserializePracticeProgressMap,
  emptyPracticeProgress,
  practiceAudioSegments,
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

test("match pair choices keep the same ids as their Turkish pairs", () => {
  const session = buildPracticeSession(study, { mode: "match", seed: "match", maxQuestions: 5 });
  const question = session.questions.find((item) => item.activity === "match");
  assert.ok(question);
  const pairIds = new Set(question.matchPairs.map((pair) => pair.id));
  assert.equal(question.choices.length, question.matchPairs.length);
  for (const choice of question.choices) {
    assert.ok(pairIds.has(choice.id));
  }
});

test("practice speaker reads the visible match cards, not an unrelated fallback card", () => {
  const session = buildPracticeSession(study, { mode: "match", seed: "match-audio", maxQuestions: 5 });
  const question = session.questions.find((item) => item.activity === "match");
  assert.ok(question);
  assert.deepEqual(
    practiceAudioSegments(question).map((segment) => segment.text),
    question.matchPairs.map((pair) => pair.turkish)
  );
});

test("practice speaker reads visible prompts without revealing hidden answers", () => {
  const recallSession = buildPracticeSession(study, { mode: "recall", seed: "recall-audio", maxQuestions: 5 });
  const recall = recallSession.questions.find((item) => item.activity === "recall");
  assert.ok(recall);
  assert.deepEqual(practiceAudioSegments(recall, "English"), [{ text: recall.prompt, lang: "en-US" }]);
  assert.notEqual(practiceAudioSegments(recall, "English")[0].text, recall.answer);

  const sentenceSession = buildPracticeSession(study, { mode: "sentence", seed: "sentence-audio", maxQuestions: 5 });
  const sentence = sentenceSession.questions.find((item) => item.activity === "sentence");
  assert.ok(sentence);
  assert.deepEqual(practiceAudioSegments(sentence, "English"), [{ text: sentence.prompt, lang: "en-US" }]);
  assert.notEqual(practiceAudioSegments(sentence, "English")[0].text, sentence.answer);
});

test("practice speaker keeps listen and blank activities tied to the current question", () => {
  const listenSession = buildPracticeSession(study, { mode: "listen", seed: "listen-audio", maxQuestions: 5 });
  const listen = listenSession.questions.find((item) => item.activity === "listen");
  assert.ok(listen);
  assert.deepEqual(practiceAudioSegments(listen), [{ text: listen.listenText, lang: "tr-TR" }]);

  const blankSession = buildPracticeSession(study, { mode: "blank", seed: "blank-audio", maxQuestions: 5 });
  const blank = blankSession.questions.find((item) => item.activity === "blank");
  assert.ok(blank);
  assert.equal(practiceAudioSegments(blank)[0].text.includes("boşluk"), true);
  assert.equal(practiceAudioSegments(blank)[0].text.includes("_____"), false);
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

test("buildPracticeSession covers all cards via round-robin interleaving", () => {
  const session = buildPracticeSession(study, { mode: "mix", seed: "round-robin", maxQuestions: 20 });
  const coveredCardIds = new Set(session.questions.filter(q => q.activity !== "match" && q.activity !== "boss").map(q => q.cardId));
  // Since we have 4 cards and maxQuestions is 20, all 4 cards must be covered in the session.
  assert.equal(coveredCardIds.size, 4);
});

test("buildPracticeSession match mode covers all cards across multiple questions", () => {
  // Let's create a temporary study response with 7 cards to force multiple match blocks (chunks of 5)
  const study7: StudyResponse = {
    ...study,
    vocabulary_cards: [
      ...study.vocabulary_cards,
      {
        turkish: "kırmızı",
        item_type: "adjective",
        translation: "red",
        cefr_level: "A1",
        example_tr: "Kırmızı gül.",
        example_translation: "Red rose.",
        learner_note: "Color adjective.",
        tts_word: "kırmızı",
        tts_sentence: "Kırmızı gül."
      },
      {
        turkish: "yeşil",
        item_type: "adjective",
        translation: "green",
        cefr_level: "A1",
        example_tr: "Yeşil çimen.",
        example_translation: "Green grass.",
        learner_note: "Color adjective.",
        tts_word: "yeşil",
        tts_sentence: "Yeşil çimen."
      },
      {
        turkish: "sarı",
        item_type: "adjective",
        translation: "yellow",
        cefr_level: "A1",
        example_tr: "Sarı güneş.",
        example_translation: "Yellow sun.",
        learner_note: "Color adjective.",
        tts_word: "sarı",
        tts_sentence: "Sarı güneş."
      }
    ]
  };
  const session = buildPracticeSession(study7, { mode: "match", seed: "match-all", maxQuestions: 10 });
  // With 7 cards, match mode should split them into 2 questions (chunk 1: 5 cards, chunk 2: 2 cards)
  assert.equal(session.questions.length, 2);
  assert.equal(session.questions[0].activity, "match");
  assert.equal(session.questions[1].activity, "match");

  const allMatchedWords = new Set<string>();
  for (const q of session.questions) {
    for (const pair of q.matchPairs) {
      allMatchedWords.add(pair.turkish);
    }
  }
  // All 7 words must be covered across the match screens
  assert.equal(allMatchedWords.size, 7);
});
