"use client";

import {
  FileText,
  Headphones,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Send,
  Square,
  Upload
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type StudyUnit = {
  text: string;
  kind: string;
  turkish_signal: boolean;
};

type StudyResponse = {
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

type VocabularyCard = {
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Study request failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
  }

  function selectVoice() {
    return (
      voices.find((item) => item.name === selectedVoice) ??
      turkishVoices[0] ??
      voices.find((item) => item.lang.toLowerCase().startsWith("tr")) ??
      voices.find((item) => item.lang.toLowerCase().startsWith("en"))
    );
  }

  function speakTexts(texts: string[]) {
    const queue = texts.map((item) => item.trim()).filter(Boolean);
    if (!("speechSynthesis" in window) || queue.length === 0) {
      return;
    }
    window.speechSynthesis.cancel();

    const voice = selectVoice();
    let index = 0;

    const playNext = () => {
      if (index >= queue.length) {
        setSpeaking(false);
        setPaused(false);
        return;
      }
      const utterance = new SpeechSynthesisUtterance(queue[index]);
      utterance.lang = "tr-TR";
      utterance.rate = speechRate;
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

  function speak() {
    speakTexts([readableText]);
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
                    speakTexts((result?.vocabulary_cards ?? []).map((card) => card.tts_word))
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
                    speakTexts((result?.vocabulary_cards ?? []).map((card) => card.tts_sentence))
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
                    <h2>Extracted</h2>
                  </div>
                </div>
                <div className="panel-body">
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
                              onClick={() => speakTexts([card.tts_word])}
                            >
                              <Play size={16} />
                            </button>
                            <button
                              aria-label={`Play example for ${card.turkish}`}
                              className="icon-button"
                              type="button"
                              onClick={() => speakTexts([card.tts_sentence])}
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
