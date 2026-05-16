"use client";

import {
  BookOpen,
  FileText,
  Headphones,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Send,
  Square,
  Trash2,
  Upload
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  PlaybackMode,
  SAVED_LESSONS_KEY,
  SavedLesson,
  SpeechSegment,
  StudyResponse,
  createSavedLesson,
  deserializeLessons,
  exampleSegments,
  formatPair,
  serializeLessons,
  upsertLesson,
  wordSegments
} from "../lib/learning";

type HealthResponse = {
  ok: boolean;
  gemini_ready: boolean;
  model: string;
  topics: number;
  error: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const levels = ["A1", "A2", "B1", "B2", "C1", "C2"];
const targetLanguages = ["English", "Turkish", "Spanish", "French", "German", "Italian"];

export default function Home() {
  const [text, setText] = useState("Merhaba, bugün Türkçe öğreniyorum.");
  const [file, setFile] = useState<File | null>(null);
  const [level, setLevel] = useState("A1");
  const [targetLanguage, setTargetLanguage] = useState("English");
  const [result, setResult] = useState<StudyResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [speechRate, setSpeechRate] = useState(1);
  const [speaking, setSpeaking] = useState(false);
  const [paused, setPaused] = useState(false);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>("bilingual");
  const [savedLessons, setSavedLessons] = useState<SavedLesson[]>([]);
  const [lessonsLoaded, setLessonsLoaded] = useState(false);
  const [lessonTitle, setLessonTitle] = useState("");
  const [lessonSearch, setLessonSearch] = useState("");
  const [activeLessonId, setActiveLessonId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;

    const checkHealth = async () => {
      attempts += 1;
      try {
        const response = await fetch(`${API_URL}/api/health`, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setHealth(payload);
        }
      } catch (caught) {
        if (!cancelled) {
          setHealth({
            ok: false,
            gemini_ready: false,
            model: "unknown",
            topics: 0,
            error: caught instanceof Error ? caught.message : "API unavailable"
          });
        }
      }
    };

    checkHealth();
    const interval = window.setInterval(() => {
      if (attempts < 24) {
        checkHealth();
      }
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    setSavedLessons(deserializeLessons(window.localStorage.getItem(SAVED_LESSONS_KEY)));
    setLessonsLoaded(true);
  }, []);

  useEffect(() => {
    if (!lessonsLoaded) {
      return;
    }
    window.localStorage.setItem(SAVED_LESSONS_KEY, serializeLessons(savedLessons));
  }, [lessonsLoaded, savedLessons]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      return;
    }

    const loadVoices = () => {
      setVoices(window.speechSynthesis.getVoices());
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const turkishVoices = useMemo(
    () =>
      voices.filter((voice) => {
        const lang = voice.lang.toLowerCase();
        return lang.startsWith("tr") || voice.name.toLowerCase().includes("turkish");
      }),
    [voices]
  );

  const readableText = useMemo(() => {
    if (!result) {
      return text;
    }
    const listenPractice = result.note.split(/listen practice/i)[1];
    return (listenPractice || result.note).replace(/[#*_`>-]/g, " ").trim();
  }, [result, text]);

  const cardTypes = useMemo(() => {
    const types = new Set(result?.vocabulary_cards.map((card) => card.item_type) ?? []);
    return ["all", ...Array.from(types).sort()];
  }, [result]);

  const filteredCards = useMemo(() => {
    const query = search.trim().toLowerCase();
    return (result?.vocabulary_cards ?? []).filter((card) => {
      const matchesType = typeFilter === "all" || card.item_type === typeFilter;
      const haystack = [
        card.turkish,
        card.translation,
        card.example_tr,
        card.example_translation,
        card.item_type
      ]
        .join(" ")
        .toLowerCase();
      return matchesType && (!query || haystack.includes(query));
    });
  }, [result, search, typeFilter]);

  async function submitStudy(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

    const body = new FormData();
    body.append("text", text);
    body.append("level", level);
    body.append("target_language", targetLanguage);
    if (file) {
      body.append("file", file);
    }

    try {
      const response = await fetch(`${API_URL}/api/study`, {
        method: "POST",
        cache: "no-store",
        body
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Study request failed.");
      }
      setResult(payload);
      setLevel(payload.study_level);
      setActiveLessonId(null);
      setLessonTitle(defaultLessonTitle(payload));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Study request failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
  }

  function defaultLessonTitle(study: StudyResponse) {
    const source = study.source_label && study.source_label !== "direct input" ? study.source_label : "Turkish lesson";
    return `${source} · ${study.study_level}`;
  }

  function selectVoiceForLanguage(lang: string) {
    const normalized = lang.toLowerCase().split("-")[0];
    const explicitVoice = voices.find((item) => item.name === selectedVoice);
    if (explicitVoice && explicitVoice.lang.toLowerCase().startsWith(normalized)) {
      return explicitVoice;
    }
    return (
      voices.find((item) => item.lang.toLowerCase() === lang.toLowerCase()) ??
      voices.find((item) => item.lang.toLowerCase().startsWith(normalized)) ??
      explicitVoice ??
      turkishVoices[0] ??
      voices[0]
    );
  }

  function speakSegments(segments: SpeechSegment[]) {
    const queue = segments.filter((segment) => segment.text.trim());
    if (!("speechSynthesis" in window) || queue.length === 0) {
      return;
    }
    window.speechSynthesis.cancel();

    let index = 0;

    const playNext = () => {
      if (index >= queue.length) {
        setSpeaking(false);
        setPaused(false);
        return;
      }
      const segment = queue[index];
      const utterance = new SpeechSynthesisUtterance(segment.text);
      utterance.lang = segment.lang;
      utterance.rate = speechRate;
      const voice = selectVoiceForLanguage(segment.lang);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      }
      utterance.onend = () => {
        index += 1;
        playNext();
      };
      utterance.onerror = () => {
        setSpeaking(false);
        setPaused(false);
      };
      window.speechSynthesis.speak(utterance);
    };

    setSpeaking(true);
    setPaused(false);
    playNext();
  }

  function speakTexts(texts: string[]) {
    speakSegments(texts.map((item) => ({ text: item, lang: "tr-TR" })));
  }

  function speak() {
    speakTexts([readableText]);
  }

  function saveLesson() {
    if (!result) {
      return;
    }
    const baseLesson =
      activeLessonId && savedLessons.find((lesson) => lesson.id === activeLessonId)
        ? { ...savedLessons.find((lesson) => lesson.id === activeLessonId)!, result }
        : createSavedLesson(result, lessonTitle || undefined);
    const lesson = {
      ...baseLesson,
      title: (lessonTitle || baseLesson.title).trim(),
      result
    };
    setSavedLessons((current) => upsertLesson(current, lesson));
    setActiveLessonId(lesson.id);
    setLessonTitle(lesson.title);
  }

  function openLesson(lesson: SavedLesson) {
    stopSpeech();
    setResult(lesson.result);
    setLevel(lesson.result.study_level);
    setTargetLanguage(lesson.result.target_language);
    setSearch("");
    setTypeFilter("all");
    setActiveLessonId(lesson.id);
    setLessonTitle(lesson.title);
    setError("");
  }

  function renameLesson(lesson: SavedLesson, title: string) {
    const trimmed = title.trim();
    if (!trimmed) {
      return;
    }
    setSavedLessons((current) =>
      current.map((item) =>
        item.id === lesson.id ? { ...item, title: trimmed, updated_at: new Date().toISOString() } : item
      )
    );
    if (activeLessonId === lesson.id) {
      setLessonTitle(trimmed);
    }
  }

  function deleteLesson(lesson: SavedLesson) {
    if (!window.confirm(`Delete saved lesson "${lesson.title}"?`)) {
      return;
    }
    setSavedLessons((current) => current.filter((item) => item.id !== lesson.id));
    if (activeLessonId === lesson.id) {
      setActiveLessonId(null);
    }
  }

  function pauseOrResume() {
    if (!("speechSynthesis" in window)) {
      return;
    }
    if (paused) {
      window.speechSynthesis.resume();
      setPaused(false);
    } else {
      window.speechSynthesis.pause();
      setPaused(true);
    }
  }

  function stopSpeech() {
    if (!("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    setSpeaking(false);
    setPaused(false);
  }

  const filteredLessons = useMemo(() => {
    const query = lessonSearch.trim().toLowerCase();
    if (!query) {
      return savedLessons;
    }
    return savedLessons.filter((lesson) =>
      [
        lesson.title,
        lesson.result.source_label,
        lesson.result.preview,
        lesson.result.target_language,
        lesson.result.study_level
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  }, [savedLessons, lessonSearch]);

  const activeLesson = activeLessonId
    ? savedLessons.find((lesson) => lesson.id === activeLessonId)
    : null;

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">TR</div>
          <div>
            <h1>Turkce Hoca</h1>
            <p>CEFR-aware Turkish tutor workspace</p>
          </div>
        </div>
        <div className="status">
          <span className={`status-dot ${health?.gemini_ready ? "ready" : ""}`} />
          <span>
            {health?.gemini_ready ? "Gemini ready" : "API waiting"} · {health?.topics ?? 0} topics
          </span>
        </div>
      </header>

      <section className="workspace">
        <form className="panel input-panel" onSubmit={submitStudy}>
          <div className="panel-header">
            <div className="panel-title">
              <FileText size={18} />
              <h2>Input</h2>
            </div>
          </div>
          <div className="panel-body">
            <div className="field">
              <label htmlFor="text-input">Text</label>
              <textarea
                id="text-input"
                value={text}
                onChange={(event) => setText(event.target.value)}
              />
            </div>

            <div className="field">
              <label htmlFor="file-input">File</label>
              <input
                id="file-input"
                type="file"
                accept=".txt,.md,.csv,.tsv,.json,.srt,.pdf,.docx,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff"
                onChange={handleFile}
              />
            </div>

            <div className="field">
              <label>Level</label>
              <div className="segmented">
                {levels.map((item) => (
                  <button
                    className={item === level ? "active" : ""}
                    key={item}
                    type="button"
                    onClick={() => setLevel(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-grid">
              <div className="field">
                <label htmlFor="target-language">Target</label>
                <select
                  id="target-language"
                  value={targetLanguage}
                  onChange={(event) => setTargetLanguage(event.target.value)}
                >
                  {targetLanguages.map((language) => (
                    <option key={language}>{language}</option>
                  ))}
                </select>
              </div>
              <button className="primary-button" disabled={loading} type="submit">
                {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                Analyze
              </button>
            </div>

            {error ? <div className="error">{error}</div> : null}
          </div>
        </form>

        <div className="result-stack">
          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <BookOpen size={18} />
                <h2>Saved Lessons</h2>
              </div>
              <strong>{savedLessons.length}</strong>
            </div>
            <div className="panel-body">
              <div className="lesson-save-row">
                <input
                  aria-label="Lesson title"
                  disabled={!result}
                  placeholder="Lesson title"
                  type="text"
                  value={lessonTitle}
                  onChange={(event) => setLessonTitle(event.target.value)}
                />
                <button className="ghost-button" disabled={!result} type="button" onClick={saveLesson}>
                  {activeLessonId ? "Update" : "Save"}
                </button>
              </div>
              <div className="search-box lesson-search">
                <Search size={16} />
                <input
                  aria-label="Search saved lessons"
                  placeholder="Search saved lessons"
                  value={lessonSearch}
                  onChange={(event) => setLessonSearch(event.target.value)}
                />
              </div>
              {filteredLessons.length ? (
                <div className="lesson-list">
                  {filteredLessons.map((lesson) => (
                    <article
                      className={`lesson-card ${lesson.id === activeLessonId ? "active" : ""}`}
                      key={lesson.id}
                    >
                      <button className="lesson-open" type="button" onClick={() => openLesson(lesson)}>
                        <strong>{lesson.title}</strong>
                        <span>
                          {lesson.result.study_level} · {lesson.result.target_language} ·{" "}
                          {new Date(lesson.created_at).toLocaleDateString()}
                        </span>
                      </button>
                      <div className="lesson-actions">
                        <button
                          className="icon-button"
                          type="button"
                          aria-label={`Rename ${lesson.title}`}
                          onClick={() => {
                            const title = window.prompt("Rename lesson", lesson.title);
                            if (title !== null) {
                              renameLesson(lesson, title);
                            }
                          }}
                        >
                          <RefreshCw size={16} />
                        </button>
                        <button
                          className="icon-button"
                          type="button"
                          aria-label={`Delete ${lesson.title}`}
                          onClick={() => deleteLesson(lesson)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="muted-copy">Saved lessons stay in this browser for later revision.</p>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <Headphones size={18} />
                <h2>Read Aloud</h2>
              </div>
            </div>
            <div className="panel-body">
              <div className="tts-grid">
                <div className="field">
                  <label htmlFor="voice-select">Voice</label>
                  <select
                    id="voice-select"
                    value={selectedVoice}
                    onChange={(event) => setSelectedVoice(event.target.value)}
                  >
                    <option value="">Auto</option>
                    {(turkishVoices.length ? turkishVoices : voices).map((voice) => (
                      <option key={`${voice.name}-${voice.lang}`} value={voice.name}>
                        {voice.name} · {voice.lang}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="playback-mode">Playback</label>
                  <select
                    id="playback-mode"
                    value={playbackMode}
                    onChange={(event) => setPlaybackMode(event.target.value as PlaybackMode)}
                  >
                    <option value="bilingual">Turkish + translation</option>
                    <option value="turkish">Turkish only</option>
                    <option value="translation">Translation only</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="speech-rate">Rate</label>
                  <div className="range-row">
                    <input
                      id="speech-rate"
                      max="1.6"
                      min="0.7"
                      step="0.1"
                      type="range"
                      value={speechRate}
                      onChange={(event) => setSpeechRate(Number(event.target.value))}
                    />
                    <strong>{speechRate.toFixed(1)}x</strong>
                  </div>
                </div>
              </div>
              <div className="player-buttons">
                <button className="ghost-button" type="button" onClick={speak}>
                  <Play size={18} />
                  Play
                </button>
                <button
                  className="ghost-button"
                  disabled={!result?.vocabulary_cards.length}
                  type="button"
                  onClick={() =>
                    speakSegments(
                      (result?.vocabulary_cards ?? []).flatMap((card) =>
                        wordSegments(card, result?.target_language ?? targetLanguage, playbackMode)
                      )
                    )
                  }
                >
                  <Play size={18} />
                  Words
                </button>
                <button
                  className="ghost-button"
                  disabled={!result?.vocabulary_cards.length}
                  type="button"
                  onClick={() =>
                    speakSegments(
                      (result?.vocabulary_cards ?? []).flatMap((card) =>
                        exampleSegments(card, result?.target_language ?? targetLanguage, playbackMode)
                      )
                    )
                  }
                >
                  <Play size={18} />
                  Examples
                </button>
                <button
                  className="ghost-button"
                  disabled={!speaking}
                  type="button"
                  onClick={pauseOrResume}
                >
                  {paused ? <RefreshCw size={18} /> : <Pause size={18} />}
                  {paused ? "Resume" : "Pause"}
                </button>
                <button
                  aria-label="Stop playback"
                  className="icon-button"
                  disabled={!speaking}
                  type="button"
                  onClick={stopSpeech}
                >
                  <Square size={18} />
                </button>
              </div>
              {!turkishVoices.length ? (
                <p className="voice-warning">
                  No Turkish browser voice is currently available. Playback will use the closest installed voice.
                </p>
              ) : null}
            </div>
          </section>

          {!result ? (
            <section className="panel empty-state">
              <div>
                <Upload size={34} />
                <p>Study output will appear here.</p>
              </div>
            </section>
          ) : (
            <>
              <section className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <FileText size={18} />
                    <h2>{activeLesson ? "Saved Lesson" : "Extracted"}</h2>
                  </div>
                </div>
                <div className="panel-body">
                  {activeLesson ? (
                    <div className="revision-banner">
                      Revising saved lesson: <strong>{activeLesson.title}</strong>
                    </div>
                  ) : null}
                  <div className="meta-grid">
                    <div className="meta-item">
                      <span>Source</span>
                      <strong>{result.source_type}</strong>
                    </div>
                    <div className="meta-item">
                      <span>Inferred</span>
                      <strong>{result.inferred_level}</strong>
                    </div>
                    <div className="meta-item">
                      <span>Studying</span>
                      <strong>{result.study_level}</strong>
                    </div>
                    <div className="meta-item">
                      <span>Target</span>
                      <strong>{result.target_language}</strong>
                    </div>
                  </div>
                  <div className="field" style={{ marginTop: 16 }}>
                    <label>Preview</label>
                    <div className="preview">{result.preview}</div>
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <FileText size={18} />
                    <h2>Vocabulary Cards</h2>
                  </div>
                  <strong>{filteredCards.length}/{result.vocabulary_cards.length}</strong>
                </div>
                <div className="panel-body">
                  {result.vocabulary_warning ? (
                    <div className="warning">{result.vocabulary_warning}</div>
                  ) : null}
                  <div className="filters">
                    <div className="search-box">
                      <Search size={16} />
                      <input
                        aria-label="Search vocabulary"
                        placeholder="Search words, translations, examples"
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                      />
                    </div>
                    <select
                      aria-label="Filter by type"
                      value={typeFilter}
                      onChange={(event) => setTypeFilter(event.target.value)}
                    >
                      {cardTypes.map((type) => (
                        <option key={type} value={type}>
                          {type === "all" ? "All types" : type}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="vocab-list">
                    {filteredCards.map((card, index) => (
                      <article className="vocab-card" key={`${card.turkish}-${index}`}>
                        <div className="vocab-main">
                          <div>
                            <span className="pill">{card.item_type}</span>
                            <h3>{card.turkish}</h3>
                            <p>{card.translation}</p>
                          </div>
                          <div className="vocab-actions">
                            <button
                              aria-label={`Play ${card.turkish}`}
                              className="icon-button"
                              type="button"
                              title={formatPair(card.turkish, card.translation)}
                              onClick={() =>
                                speakSegments(wordSegments(card, result.target_language, playbackMode))
                              }
                            >
                              <Play size={16} />
                            </button>
                            <button
                              aria-label={`Play example for ${card.turkish}`}
                              className="icon-button"
                              type="button"
                              title={formatPair(card.example_tr, card.example_translation)}
                              onClick={() =>
                                speakSegments(exampleSegments(card, result.target_language, playbackMode))
                              }
                            >
                              <Headphones size={16} />
                            </button>
                          </div>
                        </div>
                        <div className="example-block">
                          <strong>{card.example_tr}</strong>
                          <span>{card.example_translation}</span>
                        </div>
                        <div className="card-foot">
                          <span>{card.cefr_level}</span>
                          <span>{card.learner_note}</span>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <FileText size={18} />
                    <h2>Detected Units</h2>
                  </div>
                </div>
                <div className="panel-body">
                  <div className="unit-list">
                    {result.units.slice(0, 12).map((unit, index) => (
                      <div className="unit" key={`${unit.kind}-${index}`}>
                        <span>{unit.kind}</span>
                        <p>{unit.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <FileText size={18} />
                    <h2>Study Note</h2>
                  </div>
                </div>
                <div className="panel-body">
                  <pre className="note">{result.note}</pre>
                </div>
              </section>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
