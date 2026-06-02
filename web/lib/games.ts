import { StudyResponse, VocabularyCard, languageCode } from "./learning";

export type PracticeActivity = "match" | "listen" | "recall" | "sentence" | "blank" | "chunk" | "boss";
export type PracticeMode = "mix" | PracticeActivity;
export type MasteryLevel = "new" | "learning" | "strong" | "mastered";

export type MatchPair = {
  id: string;
  turkish: string;
  translation: string;
};

export type PracticeChoice = {
  id: string;
  text: string;
};

export type GameQuestion = {
  id: string;
  activity: PracticeActivity;
  cardId: string;
  cardIndex: number;
  title: string;
  instruction: string;
  prompt: string;
  turkish: string;
  translation: string;
  exampleTr: string;
  exampleTranslation: string;
  learnerNote: string;
  choices: PracticeChoice[];
  answer: string;
  answerParts: string[];
  matchPairs: MatchPair[];
  blankedText: string;
  listenText: string;
  listenLang: string;
};

export type PracticeSession = {
  id: string;
  title: string;
  level: string;
  topic: string;
  mode: PracticeMode;
  questions: GameQuestion[];
};

export type PracticeProgress = {
  xp: number;
  attempts: number;
  correct: number;
  missedCardIds: string[];
  masteryByCard: Record<string, MasteryLevel>;
  lastPracticedAt: string;
};

export const PRACTICE_PROGRESS_KEY = "turkce-hoca.practice-progress.v1";

type BuildOptions = {
  mode?: PracticeMode;
  seed?: string;
  maxQuestions?: number;
  progress?: PracticeProgress | null;
};

const activityTitles: Record<PracticeActivity, string> = {
  match: "Match Pairs",
  listen: "Listen & Pick",
  recall: "Translation Recall",
  sentence: "Sentence Builder",
  blank: "Fill the Blank",
  chunk: "Chunk Builder",
  boss: "Boss Review"
};

function clean(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function normalizeAnswer(text: string): string {
  return clean(text)
    .toLocaleLowerCase("tr-TR")
    .replace(/[.,!?;:()[\]{}'"`´’‘“”]+/g, "")
    .replace(/\s+/g, " ");
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed: string): () => number {
  let value = hashString(seed) || 1;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const next = [...items];
  for (let index = next.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [next[index], next[swapIndex]] = [next[swapIndex], next[index]];
  }
  return next;
}

function cardId(card: VocabularyCard, index: number): string {
  return `${index}-${normalizeAnswer(card.turkish).replace(/\s+/g, "-") || "card"}`;
}

function practiceCards(study: StudyResponse): Array<{ card: VocabularyCard; id: string; index: number }> {
  const seen = new Set<string>();
  return study.vocabulary_cards
    .map((card, index) => ({ card, id: cardId(card, index), index }))
    .filter(({ card }) => clean(card.turkish) && clean(card.translation))
    .filter(({ card }) => {
      const key = `${normalizeAnswer(card.turkish)}:${normalizeAnswer(card.translation)}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function choicesFor(
  correct: string,
  alternatives: string[],
  random: () => number,
  count = 4
): PracticeChoice[] {
  const normalizedCorrect = normalizeAnswer(correct);
  const distractors = shuffle(
    alternatives.filter((item) => clean(item) && normalizeAnswer(item) !== normalizedCorrect),
    random
  ).slice(0, Math.max(0, count - 1));
  return shuffle([correct, ...distractors], random).map((text, index) => ({
    id: `${index}-${normalizeAnswer(text).replace(/\s+/g, "-") || "choice"}`,
    text
  }));
}

function tokenizeSentence(text: string): string[] {
  return clean(text)
    .replace(/[.!?…]+$/g, "")
    .split(/\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function chunkTurkish(card: VocabularyCard): string[] {
  const item = clean(card.turkish);
  if (!item) {
    return [];
  }
  if (item.includes(" ")) {
    return item.split(/\s+/).filter(Boolean);
  }
  const lower = item.toLocaleLowerCase("tr-TR");
  if ((lower.endsWith("mak") || lower.endsWith("mek")) && item.length > 4) {
    return [item.slice(0, -3), item.slice(-3)];
  }
  const suffixMatch = item.match(/^(.{3,}?)(lar|ler|da|de|dan|den|ya|ye|yi|yı|yu|yü|im|ım|um|üm)$/i);
  if (suffixMatch) {
    return [suffixMatch[1], suffixMatch[2]];
  }
  return [];
}

function blankExample(card: VocabularyCard): { blankedText: string; answer: string } | null {
  const example = clean(card.example_tr);
  const answer = clean(card.turkish);
  if (!example || !answer) {
    return null;
  }
  const escaped = answer.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const phrasePattern = new RegExp(`(^|\\s)(${escaped})(?=\\s|[.,!?;:]|$)`, "iu");
  if (phrasePattern.test(example)) {
    return {
      blankedText: example.replace(phrasePattern, (_match, prefix) => `${prefix}_____`),
      answer
    };
  }
  const tokens = tokenizeSentence(example);
  const candidate = tokens.find((token) => token.length > 2 && !/^(bir|bu|ve|ile)$/i.test(token));
  if (!candidate) {
    return null;
  }
  return {
    blankedText: example.replace(candidate, "_____"),
    answer: candidate
  };
}

function baseQuestion(
  activity: PracticeActivity,
  card: VocabularyCard,
  cardIndex: number,
  id: string
): GameQuestion {
  return {
    id,
    activity,
    cardId: cardId(card, cardIndex),
    cardIndex,
    title: activityTitles[activity],
    instruction: "",
    prompt: "",
    turkish: clean(card.turkish),
    translation: clean(card.translation),
    exampleTr: clean(card.example_tr),
    exampleTranslation: clean(card.example_translation),
    learnerNote: clean(card.learner_note),
    choices: [],
    answer: "",
    answerParts: [],
    matchPairs: [],
    blankedText: "",
    listenText: clean(card.tts_word || card.turkish),
    listenLang: "tr-TR"
  };
}

function buildMatchQuestion(cards: ReturnType<typeof practiceCards>, random: () => number): GameQuestion | null {
  const pairs = shuffle(cards, random).slice(0, Math.min(5, cards.length)).map(({ card, id }) => ({
    id,
    turkish: clean(card.turkish),
    translation: clean(card.translation)
  }));
  if (pairs.length < 2) {
    return null;
  }
  return {
    ...baseQuestion("match", cards[0].card, cards[0].index, "match-pairs"),
    instruction: "Match every Turkish item to its meaning.",
    prompt: "Tap a Turkish card, then tap its translation.",
    matchPairs: pairs,
    choices: shuffle(
      pairs.map((pair) => ({ id: pair.id, text: pair.translation })),
      random
    ),
    answer: `${pairs.length} pairs`
  };
}

function buildCardQuestion(
  activity: PracticeActivity,
  card: VocabularyCard,
  cardIndex: number,
  alternatives: VocabularyCard[],
  random: () => number,
  serial: number
): GameQuestion | null {
  const question = baseQuestion(activity, card, cardIndex, `${activity}-${serial}-${cardId(card, cardIndex)}`);
  const translations = alternatives.map((item) => clean(item.translation));
  if (activity === "listen") {
    question.instruction = "Listen to Turkish, then choose the meaning.";
    question.prompt = "What did you hear?";
    question.choices = choicesFor(card.translation, translations, random);
    question.answer = clean(card.translation);
    return question.choices.length >= 2 ? question : null;
  }
  if (activity === "recall") {
    question.instruction = "Choose the Turkish for this meaning.";
    question.prompt = clean(card.translation);
    question.choices = choicesFor(card.turkish, alternatives.map((item) => clean(item.turkish)), random);
    question.answer = clean(card.turkish);
    return question.choices.length >= 2 ? question : null;
  }
  if (activity === "sentence") {
    const parts = tokenizeSentence(card.example_tr);
    if (parts.length < 2) {
      return null;
    }
    question.instruction = "Build the Turkish sentence in the correct order.";
    question.prompt = clean(card.example_translation || card.translation);
    question.answerParts = parts;
    question.choices = shuffle(parts, random).map((part, index) => ({ id: `${index}-${part}`, text: part }));
    question.answer = parts.join(" ");
    return question;
  }
  if (activity === "blank") {
    const blank = blankExample(card);
    if (!blank) {
      return null;
    }
    question.instruction = "Fill the blank with the missing Turkish word or phrase.";
    question.prompt = blank.blankedText;
    question.blankedText = blank.blankedText;
    question.answer = blank.answer;
    question.choices = choicesFor(blank.answer, alternatives.map((item) => clean(item.turkish)), random);
    return question.choices.length >= 2 ? question : null;
  }
  if (activity === "chunk") {
    const parts = chunkTurkish(card);
    if (parts.length < 2) {
      return null;
    }
    question.instruction = "Build the Turkish chunk from its pieces.";
    question.prompt = clean(card.translation);
    question.answerParts = parts;
    question.choices = shuffle(parts, random).map((part, index) => ({ id: `${index}-${part}`, text: part }));
    question.answer = parts.join(" ");
    return question;
  }
  if (activity === "boss") {
    question.instruction = "Boss Review: answer a card that needs extra attention.";
    question.prompt = clean(card.translation);
    question.choices = choicesFor(card.turkish, alternatives.map((item) => clean(item.turkish)), random);
    question.answer = clean(card.turkish);
    return question.choices.length >= 2 ? question : null;
  }
  return null;
}

function sessionTitle(study: StudyResponse): string {
  const source = study.source_label && study.source_label !== "direct input" ? study.source_label : "Current words";
  return `${source} practice`;
}

function sessionTopic(study: StudyResponse): string {
  return clean(study.textbook_sections?.[0]?.topic || study.textbook_sections?.[0]?.title || study.preview).slice(0, 120);
}

export function buildPracticeSession(study: StudyResponse, options: BuildOptions = {}): PracticeSession {
  const mode = options.mode ?? "mix";
  const maxQuestions = Math.max(1, options.maxQuestions ?? 28);
  const cards = practiceCards(study);
  const random = seededRandom(options.seed ?? `${study.source_label}:${study.study_level}:${cards.length}:${mode}`);
  const shuffledCards = shuffle(cards, random);
  const questions: GameQuestion[] = [];
  const alternatives = cards.map((item) => item.card);

  if ((mode === "mix" || mode === "match") && cards.length >= 2) {
    const match = buildMatchQuestion(cards, random);
    if (match) {
      questions.push(match);
    }
  }

  const cycle: PracticeActivity[] = mode === "mix" ? ["listen", "recall", "sentence", "blank", "chunk"] : [mode as PracticeActivity];
  let serial = 0;
  for (const { card, index } of shuffledCards) {
    for (const activity of cycle) {
      if (activity === "match" || activity === "boss") {
        continue;
      }
      const question = buildCardQuestion(activity, card, index, alternatives, random, serial);
      serial += 1;
      if (question) {
        questions.push(question);
      }
      if (questions.length >= maxQuestions) {
        break;
      }
    }
    if (questions.length >= maxQuestions) {
      break;
    }
  }

  const missed = new Set(options.progress?.missedCardIds ?? []);
  const lowMastery = shuffledCards.filter(({ id }) => missed.has(id) || !["strong", "mastered"].includes(options.progress?.masteryByCard[id] ?? "new"));
  for (const { card, index } of lowMastery.slice(0, 4)) {
    if (mode !== "mix" && mode !== "boss") {
      break;
    }
    const question = buildCardQuestion("boss", card, index, alternatives, random, serial);
    serial += 1;
    if (question) {
      questions.push(question);
    }
  }

  return {
    id: `practice-${hashString(`${study.source_label}:${study.preview}:${mode}:${questions.length}`).toString(16)}`,
    title: sessionTitle(study),
    level: study.study_level,
    topic: sessionTopic(study),
    mode,
    questions: questions.slice(0, maxQuestions)
  };
}

export function emptyPracticeProgress(): PracticeProgress {
  return {
    xp: 0,
    attempts: 0,
    correct: 0,
    missedCardIds: [],
    masteryByCard: {},
    lastPracticedAt: ""
  };
}

export function normalizePracticeProgress(value: unknown): PracticeProgress {
  if (!value || typeof value !== "object") {
    return emptyPracticeProgress();
  }
  const progress = value as Partial<PracticeProgress>;
  return {
    xp: Number.isFinite(progress.xp) ? Math.max(0, Math.round(progress.xp ?? 0)) : 0,
    attempts: Number.isFinite(progress.attempts) ? Math.max(0, Math.round(progress.attempts ?? 0)) : 0,
    correct: Number.isFinite(progress.correct) ? Math.max(0, Math.round(progress.correct ?? 0)) : 0,
    missedCardIds: Array.isArray(progress.missedCardIds) ? progress.missedCardIds.filter((item): item is string => typeof item === "string") : [],
    masteryByCard:
      progress.masteryByCard && typeof progress.masteryByCard === "object"
        ? Object.fromEntries(
            Object.entries(progress.masteryByCard).filter((entry): entry is [string, MasteryLevel] =>
              ["new", "learning", "strong", "mastered"].includes(entry[1] as string)
            )
          )
        : {},
    lastPracticedAt: typeof progress.lastPracticedAt === "string" ? progress.lastPracticedAt : ""
  };
}

export function applyPracticeAnswer(progress: PracticeProgress, question: GameQuestion, correct: boolean, firstTry = true): PracticeProgress {
  const current = normalizePracticeProgress(progress);
  const mastery = current.masteryByCard[question.cardId] ?? "new";
  const nextMastery: MasteryLevel = correct
    ? mastery === "new"
      ? "learning"
      : mastery === "learning"
        ? "strong"
        : "mastered"
    : "learning";
  const missed = new Set(current.missedCardIds);
  if (correct) {
    missed.delete(question.cardId);
  } else {
    missed.add(question.cardId);
  }
  return {
    xp: current.xp + (correct ? (firstTry ? 10 : 6) : 1),
    attempts: current.attempts + 1,
    correct: current.correct + (correct ? 1 : 0),
    missedCardIds: Array.from(missed),
    masteryByCard: {
      ...current.masteryByCard,
      [question.cardId]: nextMastery
    },
    lastPracticedAt: new Date().toISOString()
  };
}

export function answerMatches(question: GameQuestion, answer: string | string[]): boolean {
  if (question.activity === "match") {
    return false;
  }
  if (Array.isArray(answer)) {
    return normalizeAnswer(answer.join(" ")) === normalizeAnswer(question.answerParts.join(" ") || question.answer);
  }
  return normalizeAnswer(answer) === normalizeAnswer(question.answer);
}

export function serializePracticeProgressMap(progress: Record<string, PracticeProgress>): string {
  return JSON.stringify(progress);
}

export function deserializePracticeProgressMap(raw: string | null): Record<string, PracticeProgress> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).map(([key, value]) => [key, normalizePracticeProgress(value)])
    );
  } catch {
    return {};
  }
}

export function progressAccuracy(progress: PracticeProgress): number {
  return progress.attempts > 0 ? Math.round((progress.correct / progress.attempts) * 100) : 0;
}

export function practiceListenSegment(question: GameQuestion) {
  return {
    text: question.listenText || question.turkish,
    lang: question.listenLang || languageCode("Turkish")
  };
}
