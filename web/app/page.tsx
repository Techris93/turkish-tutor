"use client";

import {
  BookOpen,
  FileText,
  Headphones,
  KeyRound,
  Loader2,
  LogIn,
  LogOut,
  Pause,
  Play,
  RefreshCw,
  Search,
  Send,
  SkipBack,
  SkipForward,
  Square,
  Trash2,
  UserPlus,
  Upload
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  PlaybackEngine,
  PlaybackMode,
  PlaybackQueueItem,
  SAVED_LESSONS_KEY,
  SavedLesson,
  SpeechSegment,
  StudyResponse,
  audioCacheKey,
  createSavedLesson,
  deserializeLessons,
  examplePlaybackQueue,
  exampleSegments,
  formatPair,
  normalizePlaybackRate,
  playbackProgress,
  serializeLessons,
  shouldUseGeneratedAudio,
  textQueueItem,
  upsertLesson,
  wordPlaybackQueue,
  wordSegments
} from "../lib/learning";
import {
  REMEMBERED_SESSION_TOKEN_KEY,
  SESSION_TOKEN_KEY,
  consumeOAuthRememberPreference,
  getStoredSessionToken,
  storeOAuthRememberPreference,
  storeSessionToken as persistSessionToken
} from "../lib/authTokens";

type HealthResponse = {
  ok: boolean;
  gemini_ready: boolean;
  model: string;
  topics: number;
  error: string;
};

type TTSConfigResponse = {
  provider: string;
  configured: boolean;
  auth_required: boolean;
  voices: string[];
  model: string;
};

type AuthUser = {
  id: string;
  email: string;
  name: string;
  created_at: string;
};

type AuthResponse = {
  user: AuthUser;
  session_token?: string | null;
};

type AuthMode = "login" | "signup" | "reset";

type OAuthProvider = {
  provider: string;
  configured: boolean;
  authorization_url: string | null;
};

type OAuthRedeemResponse = {
  user: AuthUser;
  lessons: SavedLesson[];
  session_token: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const ALLOW_REMEMBER_ME = process.env.NEXT_PUBLIC_ALLOW_REMEMBER_ME !== "false";
const levels = ["A1", "A2", "B1", "B2", "C1", "C2"];
const targetLanguages = ["English", "Turkish", "Spanish", "French", "German", "Italian"];

function getSessionToken() {
  if (typeof window === "undefined") {
    return "";
  }
  if (!ALLOW_REMEMBER_ME) {
    window.localStorage.removeItem(REMEMBERED_SESSION_TOKEN_KEY);
    return window.sessionStorage.getItem(SESSION_TOKEN_KEY) ?? "";
  }
  return getStoredSessionToken(window.sessionStorage, window.localStorage);
}

function storeSessionToken(token?: string | null, remember = false) {
  if (typeof window === "undefined") {
    return;
  }
  persistSessionToken(window.sessionStorage, window.localStorage, token, remember && ALLOW_REMEMBER_ME);
}

function authHeaders(): Record<string, string> {
  const token = getSessionToken();
  return token ? { "X-Session-Token": token } : {};
}

async function apiJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const token = getSessionToken();
  if (token) {
    headers.set("X-Session-Token", token);
  }

  const response = await fetch(`${API_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...options,
    headers
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "Request failed.");
  }
  return payload as T;
}

function isAuthRequired(error: unknown): boolean {
  return error instanceof Error && /authentication required|session expired/i.test(error.message);
}

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
  const [playbackLabel, setPlaybackLabel] = useState("");
  const [playbackCurrent, setPlaybackCurrent] = useState(0);
  const [playbackTotal, setPlaybackTotal] = useState(0);
  const [playbackNotice, setPlaybackNotice] = useState("");
  const [pwaReady, setPwaReady] = useState(false);
  const [playbackEngine, setPlaybackEngine] = useState<PlaybackEngine>("browser");
  const [ttsConfig, setTtsConfig] = useState<TTSConfigResponse | null>(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [activeEngine, setActiveEngine] = useState<"generated" | "browser" | "idle">("idle");
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>("bilingual");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [resetToken, setResetToken] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([]);
  const [savedLessons, setSavedLessons] = useState<SavedLesson[]>([]);
  const [localLessons, setLocalLessons] = useState<SavedLesson[]>([]);
  const [lessonsLoaded, setLessonsLoaded] = useState(false);
  const [lessonsLoading, setLessonsLoading] = useState(false);
  const [lessonError, setLessonError] = useState("");
  const [lessonTitle, setLessonTitle] = useState("");
  const [lessonSearch, setLessonSearch] = useState("");
  const [activeLessonId, setActiveLessonId] = useState<string | null>(null);
  const playbackQueueRef = useRef<PlaybackQueueItem[]>([]);
  const playbackItemIndexRef = useRef(0);
  const playbackSegmentIndexRef = useRef(0);
  const playbackRunRef = useRef(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCacheRef = useRef<Map<string, string>>(new Map());
  const speechRateRef = useRef(1);

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
    const local = deserializeLessons(window.localStorage.getItem(SAVED_LESSONS_KEY));
    setLocalLessons(local);
    setSavedLessons(local);
    setLessonsLoaded(true);
    const params = new URLSearchParams(window.location.search);
    const token = params.get("reset_token");
    const oauth = params.get("oauth");
    const handoff = params.get("handoff");
    if (token) {
      setResetToken(token);
      setAuthMode("reset");
      setAuthMessage("Choose a new password to finish resetting your account.");
    } else if (oauth === "success") {
      setAuthMessage("Finishing sign-in...");
      void (async () => {
        const rememberOAuth = ALLOW_REMEMBER_ME && consumeOAuthRememberPreference(window.localStorage);
        try {
          const payload = handoff
            ? await apiJson<OAuthRedeemResponse>("/api/auth/oauth/redeem", {
                method: "POST",
                body: JSON.stringify({ handoff })
              })
            : {
                user: (await apiJson<AuthResponse>("/api/auth/me")).user,
                lessons: await apiJson<SavedLesson[]>("/api/lessons"),
                session_token: null
              };
          storeSessionToken(payload.session_token, rememberOAuth);
          setUser(payload.user);
          setSavedLessons(payload.lessons);
          setActiveLessonId(null);
          setAuthError("");
          setAuthMessage(rememberOAuth ? "Signed in with OAuth. This device will remember you." : "Signed in with OAuth.");
        } catch {
          setUser(null);
          setAuthMessage("");
          setAuthError(
            "OAuth finished, but this browser did not accept the session. Try again, or use email login."
          );
        } finally {
          window.history.replaceState({}, "", window.location.pathname);
        }
      })();
    } else if (oauth === "error") {
      consumeOAuthRememberPreference(window.localStorage);
      const reason = params.get("reason");
      setAuthError(
        reason === "account_mismatch"
          ? "That Google/GitHub email belongs to a different account. Log out first to switch accounts, or use OAuth with the same email."
          : "OAuth login could not be completed. Please try again or use email login."
      );
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []);

  useEffect(() => {
    if (!("serviceWorker" in navigator)) {
      return;
    }
    const canRegister = window.location.protocol === "https:" || window.location.hostname === "localhost";
    if (!canRegister) {
      return;
    }
    navigator.serviceWorker
      .register("/sw.js")
      .then(() => setPwaReady(true))
      .catch(() => setPwaReady(false));
  }, []);

  useEffect(() => {
    if (!lessonsLoaded) {
      return;
    }
    window.localStorage.setItem(SAVED_LESSONS_KEY, serializeLessons(localLessons));
  }, [lessonsLoaded, localLessons]);

  useEffect(() => {
    let cancelled = false;
    async function loadAccount() {
      try {
        const payload = await apiJson<AuthResponse>("/api/auth/me");
        if (!cancelled) {
          setUser(payload.user);
          const lessons = await apiJson<SavedLesson[]>("/api/lessons");
          setSavedLessons(lessons);
          setActiveLessonId(null);
        }
      } catch {
        if (!cancelled) {
          setUser(null);
        }
      }
    }
    loadAccount();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadOAuth() {
      try {
        const payload = await apiJson<{ providers: OAuthProvider[] }>("/api/auth/oauth/config");
        if (!cancelled) {
          setOauthProviders(payload.providers);
        }
      } catch {
        if (!cancelled) {
          setOauthProviders([]);
        }
      }
    }
    loadOAuth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadTtsConfig() {
      try {
        const payload = await apiJson<TTSConfigResponse>("/api/tts/config");
        if (!cancelled) {
          setTtsConfig(payload);
        }
      } catch {
        if (!cancelled) {
          setTtsConfig({ provider: "none", configured: false, auth_required: true, voices: [], model: "" });
        }
      }
    }
    loadTtsConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const cache = audioCacheRef.current;
    return () => {
      for (const url of cache.values()) {
        URL.revokeObjectURL(url);
      }
      cache.clear();
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

  useEffect(() => {
    if (!("mediaSession" in navigator)) {
      return;
    }

    const handlers: Array<[MediaSessionAction, MediaSessionActionHandler]> = [
      [
        "play",
        () => {
          if (audioRef.current) {
            void audioRef.current?.play();
            setPaused(false);
            navigator.mediaSession.playbackState = "playing";
          } else if ("speechSynthesis" in window) {
            window.speechSynthesis.resume();
            setPaused(false);
            navigator.mediaSession.playbackState = "playing";
          }
        }
      ],
      ["pause", () => pauseOrResume()],
      ["stop", () => stopSpeech()],
      ["previoustrack", () => skipPlayback(-1)],
      ["nexttrack", () => skipPlayback(1)]
    ];

    for (const [action, handler] of handlers) {
      try {
        navigator.mediaSession.setActionHandler(action, handler);
      } catch {
        // Some browsers expose Media Session but not every action.
      }
    }

    return () => {
      for (const [action] of handlers) {
        try {
          navigator.mediaSession.setActionHandler(action, null);
        } catch {
          // Ignore unsupported actions during cleanup.
        }
      }
    };
    // Media Session handlers delegate to the current playback refs; registering once avoids duplicate handlers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        credentials: "include",
        body,
        headers: authHeaders()
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

  function setMediaSession(item: PlaybackQueueItem | null, state: MediaSessionPlaybackState = "none") {
    if (!("mediaSession" in navigator)) {
      return;
    }
    navigator.mediaSession.playbackState = state;
    navigator.mediaSession.metadata = item
      ? new MediaMetadata({
          title: item.title || "Turkce Hoca",
          artist: item.subtitle || "Turkish tutor",
          album: playbackLabel || "Read Aloud"
        })
      : null;
  }

  function resetPlaybackState(message = "") {
    audioRef.current?.pause();
    audioRef.current = null;
    setSpeaking(false);
    setPaused(false);
    setAudioLoading(false);
    setActiveEngine("idle");
    setPlaybackCurrent(0);
    setPlaybackTotal(0);
    setPlaybackLabel("");
    setPlaybackNotice(message);
    setMediaSession(null, "none");
  }

  function speakSegment(segment: SpeechSegment, runId: number) {
    setAudioLoading(false);
    setActiveEngine("browser");
    const utterance = new SpeechSynthesisUtterance(segment.text);
    utterance.lang = segment.lang;
    utterance.rate = speechRateRef.current;
    const voice = selectVoiceForLanguage(segment.lang);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
    utterance.onend = () => {
      if (runId !== playbackRunRef.current) {
        return;
      }
      playbackSegmentIndexRef.current += 1;
      void playCurrentSegment();
    };
    utterance.onerror = () => {
      if (runId !== playbackRunRef.current) {
        return;
      }
      resetPlaybackState("Playback stopped. This browser may have blocked background speech.");
    };
    window.speechSynthesis.speak(utterance);
  }

  async function fetchGeneratedAudio(segment: SpeechSegment): Promise<string> {
    const provider = ttsConfig?.provider || "auto";
    const voice = segment.lang.toLowerCase().startsWith("tr") ? "tr" : "default";
    const rate = speechRateRef.current;
    const key = audioCacheKey(segment, provider, voice, rate);
    const cached = audioCacheRef.current.get(key);
    if (cached) {
      return cached;
    }
    const response = await fetch(`${API_URL}/api/tts/audio`, {
      method: "POST",
      cache: "no-store",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders()
      },
      body: JSON.stringify({
        text: segment.text,
        language: segment.lang,
        speed: rate,
        provider: ttsConfig?.provider && ttsConfig.provider !== "none" ? ttsConfig.provider : undefined
      })
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Generated audio is unavailable.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    audioCacheRef.current.set(key, url);
    return url;
  }

  async function playGeneratedSegment(segment: SpeechSegment, runId: number) {
    setAudioLoading(true);
    setActiveEngine("generated");
    const url = await fetchGeneratedAudio(segment);
    if (runId !== playbackRunRef.current) {
      return;
    }
    const audio = new Audio(url);
    audio.preload = "auto";
    audio.playbackRate = speechRateRef.current;
    audioRef.current = audio;
    audio.onended = () => {
      if (runId !== playbackRunRef.current) {
        return;
      }
      playbackSegmentIndexRef.current += 1;
      void playCurrentSegment();
    };
    audio.onerror = () => {
      if (runId !== playbackRunRef.current) {
        return;
      }
      setPlaybackNotice("Generated audio failed during playback. Falling back to browser speech.");
      speakSegment(segment, runId);
    };
    setAudioLoading(false);
    await audio.play();
  }

  async function playCurrentSegment() {
    const queue = playbackQueueRef.current;
    const item = queue[playbackItemIndexRef.current];
    if (!item) {
      resetPlaybackState("");
      return;
    }

    const segment = item.segments[playbackSegmentIndexRef.current];
    if (!segment) {
      playbackItemIndexRef.current += 1;
      playbackSegmentIndexRef.current = 0;
      void playCurrentSegment();
      return;
    }

    const runId = playbackRunRef.current;
    setSpeaking(true);
    setPaused(false);
    setPlaybackCurrent(playbackItemIndexRef.current);
    setPlaybackTotal(queue.length);
    setMediaSession(item, "playing");

    const useGenerated = shouldUseGeneratedAudio(playbackEngine, Boolean(ttsConfig?.configured), Boolean(user));
    if (useGenerated) {
      try {
        await playGeneratedSegment(segment, runId);
        return;
      } catch (caught) {
        const reason = caught instanceof Error ? caught.message : "Generated audio failed.";
        if (playbackEngine === "generated") {
          resetPlaybackState(reason);
          return;
        }
        setPlaybackNotice(`${reason} Falling back to browser speech.`);
      }
    }
    speakSegment(segment, runId);
  }

  function startPlaybackQueue(queueItems: PlaybackQueueItem[], label: string) {
    const queue = queueItems.filter((item) => item.segments.length);
    const useGenerated = shouldUseGeneratedAudio(playbackEngine, Boolean(ttsConfig?.configured), Boolean(user));
    if (playbackEngine === "generated" && !useGenerated) {
      setPlaybackNotice(
        !ttsConfig?.configured
          ? "Generated audio is not configured. Choose Browser speech, or add OpenAI TTS settings on the API."
          : "Sign in to use generated audio, or choose Browser speech."
      );
      return;
    }
    if (!useGenerated && !("speechSynthesis" in window)) {
      setPlaybackNotice("This browser does not support built-in text-to-speech.");
      return;
    }
    if (!queue.length) {
      setPlaybackNotice("Nothing is ready to read aloud yet.");
      return;
    }
    playbackRunRef.current += 1;
    window.speechSynthesis?.cancel();
    audioRef.current?.pause();
    audioRef.current = null;
    playbackQueueRef.current = queue;
    playbackItemIndexRef.current = 0;
    playbackSegmentIndexRef.current = 0;
    setPlaybackLabel(label);
    setPlaybackTotal(queue.length);
    setPlaybackCurrent(0);
    setPlaybackNotice(
      useGenerated
        ? "Using generated audio. This is the recommended mode for mobile background and lock-screen listening."
        : "Using browser speech. It can continue while minimized in many browsers, but locked-screen playback depends on your device."
    );
    void playCurrentSegment();
  }

  function speakSegments(segments: SpeechSegment[], label = "Read Aloud") {
    startPlaybackQueue(
      [
        {
          id: "custom",
          title: label,
          subtitle: "Turkce Hoca",
          segments
        }
      ],
      label
    );
  }

  function speakTexts(texts: string[]) {
    startPlaybackQueue(texts.map((item) => textQueueItem(item)), "Study note");
  }

  function speak() {
    speakTexts([readableText]);
  }

  async function loadRemoteLessons() {
    setLessonsLoading(true);
    setLessonError("");
    try {
      const lessons = await apiJson<SavedLesson[]>("/api/lessons");
      setSavedLessons(lessons);
    } catch (caught) {
      if (isAuthRequired(caught)) {
        try {
          await apiJson<AuthResponse>("/api/auth/me");
          const lessons = await apiJson<SavedLesson[]>("/api/lessons");
          setSavedLessons(lessons);
          return;
        } catch {
          setUser(null);
          setSavedLessons(localLessons);
          setAuthError("Your sign-in session was not accepted. Please log in again.");
          return;
        }
      }
      setLessonError(caught instanceof Error ? caught.message : "Could not load saved lessons.");
    } finally {
      setLessonsLoading(false);
    }
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthLoading(true);
    setAuthError("");
    setAuthMessage("");
    try {
      if (authMode === "reset") {
        if (resetToken.trim()) {
          await apiJson<{ ok: boolean }>("/api/auth/password-reset/confirm", {
            method: "POST",
            body: JSON.stringify({ token: resetToken.trim(), password: authPassword })
          });
          setAuthMessage("Password updated. You can log in now.");
          setResetToken("");
          setAuthPassword("");
          setAuthMode("login");
          window.history.replaceState({}, "", window.location.pathname);
        } else {
          const payload = await apiJson<{ message: string; reset_token?: string | null }>(
            "/api/auth/password-reset/request",
            {
              method: "POST",
              body: JSON.stringify({ email: authEmail })
            }
          );
          setAuthMessage(payload.reset_token ? `${payload.message} Dev token: ${payload.reset_token}` : payload.message);
        }
        return;
      }

      const path = authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
      const payload = await apiJson<AuthResponse>(path, {
        method: "POST",
        body: JSON.stringify({
          email: authEmail,
          password: authPassword,
          name: authName
        })
      });
      storeSessionToken(payload.session_token, rememberMe);
      setUser(payload.user);
      setAuthPassword("");
      setAuthMessage(
        rememberMe
          ? authMode === "signup"
            ? "Account created. This device will remember you."
            : "Logged in. This device will remember you."
          : authMode === "signup"
            ? "Account created."
            : "Logged in."
      );
      await loadRemoteLessons();
      setActiveLessonId(null);
    } catch (caught) {
      setAuthError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setAuthLoading(false);
    }
  }

  async function logout() {
    setAuthLoading(true);
    setAuthError("");
    setAuthMessage("");
    try {
      await apiJson<{ ok: boolean }>("/api/auth/logout", { method: "POST" });
      storeSessionToken(null);
      setUser(null);
      setSavedLessons(localLessons);
      setActiveLessonId(null);
      setAuthMessage("Logged out. Local draft lessons are still available in this browser.");
    } catch (caught) {
      setAuthError(caught instanceof Error ? caught.message : "Logout failed.");
    } finally {
      setAuthLoading(false);
    }
  }

  async function importLocalLessons() {
    if (!user || localLessons.length === 0) {
      return;
    }
    setLessonsLoading(true);
    setLessonError("");
    try {
      const imported: SavedLesson[] = [];
      for (const lesson of localLessons) {
        const saved = await apiJson<SavedLesson>("/api/lessons", {
          method: "POST",
          body: JSON.stringify({ title: lesson.title, result: lesson.result })
        });
        imported.push(saved);
      }
      setLocalLessons([]);
      window.localStorage.removeItem(SAVED_LESSONS_KEY);
      await loadRemoteLessons();
      setAuthMessage(`Imported ${imported.length} local lesson${imported.length === 1 ? "" : "s"} into your account.`);
    } catch (caught) {
      setLessonError(caught instanceof Error ? caught.message : "Could not import local lessons.");
    } finally {
      setLessonsLoading(false);
    }
  }

  function startOAuth(provider: OAuthProvider) {
    if (!provider.authorization_url) {
      return;
    }
    storeOAuthRememberPreference(window.localStorage, rememberMe && ALLOW_REMEMBER_ME);
    window.location.href = provider.authorization_url;
  }

  async function saveLesson() {
    if (!result) {
      return;
    }
    const title = (lessonTitle || defaultLessonTitle(result)).trim();
    setLessonError("");
    try {
      if (user) {
        const existing = activeLessonId ? savedLessons.find((lesson) => lesson.id === activeLessonId) : null;
        const saved = existing
          ? await apiJson<SavedLesson>(`/api/lessons/${existing.id}`, {
              method: "PATCH",
              body: JSON.stringify({ title, result })
            })
          : await apiJson<SavedLesson>("/api/lessons", {
              method: "POST",
              body: JSON.stringify({ title, result })
            });
        setSavedLessons((current) => upsertLesson(current, saved));
        setActiveLessonId(saved.id);
        setLessonTitle(saved.title);
        return;
      }

      const baseLesson =
        activeLessonId && savedLessons.find((lesson) => lesson.id === activeLessonId)
          ? { ...savedLessons.find((lesson) => lesson.id === activeLessonId)!, result }
          : createSavedLesson(result, title);
      const lesson = {
        ...baseLesson,
        title,
        result
      };
      setLocalLessons((current) => upsertLesson(current, lesson));
      setSavedLessons((current) => upsertLesson(current, lesson));
      setActiveLessonId(lesson.id);
      setLessonTitle(lesson.title);
    } catch (caught) {
      setLessonError(caught instanceof Error ? caught.message : "Could not save lesson.");
    }
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

  async function renameLesson(lesson: SavedLesson, title: string) {
    const trimmed = title.trim();
    if (!trimmed) {
      return;
    }
    setLessonError("");
    try {
      if (user) {
        const saved = await apiJson<SavedLesson>(`/api/lessons/${lesson.id}`, {
          method: "PATCH",
          body: JSON.stringify({ title: trimmed })
        });
        setSavedLessons((current) => upsertLesson(current, saved));
      } else {
        const renamed = { ...lesson, title: trimmed, updated_at: new Date().toISOString() };
        setLocalLessons((current) => upsertLesson(current, renamed));
        setSavedLessons((current) => upsertLesson(current, renamed));
      }
      if (activeLessonId === lesson.id) {
        setLessonTitle(trimmed);
      }
    } catch (caught) {
      setLessonError(caught instanceof Error ? caught.message : "Could not rename lesson.");
    }
  }

  async function deleteLesson(lesson: SavedLesson) {
    if (!window.confirm(`Delete saved lesson "${lesson.title}"?`)) {
      return;
    }
    setLessonError("");
    try {
      if (user) {
        await apiJson<{ ok: boolean }>(`/api/lessons/${lesson.id}`, { method: "DELETE" });
      } else {
        setLocalLessons((current) => current.filter((item) => item.id !== lesson.id));
      }
      setSavedLessons((current) => current.filter((item) => item.id !== lesson.id));
      if (activeLessonId === lesson.id) {
        setActiveLessonId(null);
      }
    } catch (caught) {
      setLessonError(caught instanceof Error ? caught.message : "Could not delete lesson.");
    }
  }

  function pauseOrResume() {
    if (paused) {
      if (activeEngine === "generated") {
        void audioRef.current?.play();
      } else if ("speechSynthesis" in window) {
        window.speechSynthesis.resume();
      }
      setPaused(false);
      if ("mediaSession" in navigator) {
        navigator.mediaSession.playbackState = "playing";
      }
    } else {
      if (activeEngine === "generated") {
        audioRef.current?.pause();
      } else if ("speechSynthesis" in window) {
        window.speechSynthesis.pause();
      }
      setPaused(true);
      if ("mediaSession" in navigator) {
        navigator.mediaSession.playbackState = "paused";
      }
    }
  }

  function skipPlayback(direction: number) {
    const queue = playbackQueueRef.current;
    if (!queue.length) {
      return;
    }
    const nextIndex = Math.min(Math.max(playbackItemIndexRef.current + direction, 0), queue.length - 1);
    playbackRunRef.current += 1;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    audioRef.current?.pause();
    audioRef.current = null;
    playbackItemIndexRef.current = nextIndex;
    playbackSegmentIndexRef.current = 0;
    void playCurrentSegment();
  }

  function changeSpeechRate(nextRate: number) {
    const normalized = normalizePlaybackRate(nextRate);
    speechRateRef.current = normalized;
    setSpeechRate(normalized);
    if (audioRef.current) {
      audioRef.current.playbackRate = normalized;
    }

    if (
      activeEngine === "browser" &&
      speaking &&
      !paused &&
      playbackQueueRef.current.length &&
      "speechSynthesis" in window
    ) {
      playbackRunRef.current += 1;
      window.speechSynthesis.cancel();
      void playCurrentSegment();
    }
  }

  function stopSpeech() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    playbackRunRef.current += 1;
    audioRef.current?.pause();
    audioRef.current = null;
    playbackQueueRef.current = [];
    playbackItemIndexRef.current = 0;
    playbackSegmentIndexRef.current = 0;
    resetPlaybackState("");
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

            <p className="muted-copy">
              Privacy: text and uploaded files are sent to Gemini for analysis. Generated audio sends selected text to
              the configured TTS provider only when you choose Generated audio.
            </p>

            {error ? <div className="error">{error}</div> : null}
          </div>
        </form>

        <div className="result-stack">
          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                {user ? <KeyRound size={18} /> : authMode === "signup" ? <UserPlus size={18} /> : <LogIn size={18} />}
                <h2>{user ? "Account" : "Sign In"}</h2>
              </div>
            </div>
            <div className="panel-body">
              {user ? (
                <div className="account-box">
                  <div>
                    <strong>{user.name}</strong>
                    <span>{user.email}</span>
                  </div>
                  <button className="ghost-button" disabled={authLoading} type="button" onClick={logout}>
                    <LogOut size={18} />
                    Logout
                  </button>
                </div>
              ) : (
                <form className="auth-form" onSubmit={submitAuth}>
                  <div className="auth-tabs">
                    <button
                      className={authMode === "login" ? "active" : ""}
                      type="button"
                      onClick={() => setAuthMode("login")}
                    >
                      Login
                    </button>
                    <button
                      className={authMode === "signup" ? "active" : ""}
                      type="button"
                      onClick={() => setAuthMode("signup")}
                    >
                      Sign up
                    </button>
                    <button
                      className={authMode === "reset" ? "active" : ""}
                      type="button"
                      onClick={() => setAuthMode("reset")}
                    >
                      Reset
                    </button>
                  </div>

                  {authMode === "signup" ? (
                    <div className="field compact-field">
                      <label htmlFor="auth-name">Name</label>
                      <input
                        id="auth-name"
                        type="text"
                        autoComplete="name"
                        value={authName}
                        onChange={(event) => setAuthName(event.target.value)}
                      />
                    </div>
                  ) : null}

                  <div className="auth-grid">
                    <div className="field compact-field">
                      <label htmlFor="auth-email">Email</label>
                      <input
                        id="auth-email"
                        type="email"
                        autoComplete="email"
                        value={authEmail}
                        onChange={(event) => setAuthEmail(event.target.value)}
                      />
                    </div>
                    <div className="field compact-field">
                      <label htmlFor="auth-password">
                        {authMode === "reset" && resetToken ? "New password" : "Password"}
                      </label>
                      <input
                        id="auth-password"
                        type="password"
                        autoComplete={authMode === "login" ? "current-password" : "new-password"}
                        value={authPassword}
                        onChange={(event) => setAuthPassword(event.target.value)}
                      />
                    </div>
                  </div>

                  {authMode === "reset" ? (
                    <div className="field compact-field">
                      <label htmlFor="reset-token">Reset token</label>
                      <input
                        id="reset-token"
                        type="text"
                        placeholder="Paste token after email delivery is configured"
                        value={resetToken}
                        onChange={(event) => setResetToken(event.target.value)}
                      />
                    </div>
                  ) : null}

                  {authMode !== "reset" && ALLOW_REMEMBER_ME ? (
                    <label className="check-row" htmlFor="remember-me">
                      <input
                        checked={rememberMe}
                        id="remember-me"
                        type="checkbox"
                        onChange={(event) => setRememberMe(event.target.checked)}
                      />
                      <span>
                        <strong>Remember me</strong>
                        <small>Only use this on your own device.</small>
                      </span>
                    </label>
                  ) : null}
                  {authMode !== "reset" && !ALLOW_REMEMBER_ME ? (
                    <p className="muted-copy">Remember me is disabled for this deployment; sign-in is session-only.</p>
                  ) : null}

                  <button className="primary-button" disabled={authLoading} type="submit">
                    {authLoading ? <Loader2 className="spin" size={18} /> : <KeyRound size={18} />}
                    {authMode === "signup" ? "Create account" : authMode === "reset" && !resetToken ? "Request reset" : "Continue"}
                  </button>
                </form>
              )}

              {oauthProviders.length ? (
                <div className="oauth-row">
                  {oauthProviders.map((provider) => (
                    <button
                      className="ghost-button"
                      disabled={!provider.authorization_url}
                      key={provider.provider}
                      type="button"
                      onClick={() => startOAuth(provider)}
                      title={
                        provider.authorization_url
                          ? `Continue with ${provider.provider}.`
                          : provider.configured
                            ? `${provider.provider} credentials are configured; callback routes still need to be wired.`
                          : `${provider.provider} OAuth is not configured yet.`
                      }
                    >
                      {provider.provider} OAuth
                    </button>
                  ))}
                </div>
              ) : null}

              {authError ? <div className="error">{authError}</div> : null}
              {authMessage ? <div className="success">{authMessage}</div> : null}
              {!user ? (
                <p className="muted-copy">
                  Sign in to sync saved lessons across refreshes and devices. Local lessons stay available until you import them.
                </p>
              ) : null}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <BookOpen size={18} />
                <h2>Saved Lessons</h2>
              </div>
              <strong>{lessonsLoading ? "..." : savedLessons.length}</strong>
            </div>
            <div className="panel-body">
              {user && localLessons.length ? (
                <div className="revision-banner">
                  You have {localLessons.length} local lesson{localLessons.length === 1 ? "" : "s"} from this browser.
                  <button className="inline-button" disabled={lessonsLoading} type="button" onClick={importLocalLessons}>
                    Import to account
                  </button>
                </div>
              ) : null}
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
              {lessonError ? <div className="error">{lessonError}</div> : null}
              {!user ? (
                <p className="muted-copy">Not signed in: lessons are saved only in this browser until you log in.</p>
              ) : null}
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
                <p className="muted-copy">
                  {user
                    ? "Saved lessons from your account will appear here."
                    : "Local saved lessons stay in this browser. Sign in to keep them across devices."}
                </p>
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
                    <option value="bilingual">Translation + Turkish</option>
                    <option value="turkish">Turkish only</option>
                    <option value="translation">Translation only</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="playback-engine">Engine</label>
                  <select
                    id="playback-engine"
                    value={playbackEngine}
                    onChange={(event) => setPlaybackEngine(event.target.value as PlaybackEngine)}
                  >
                    <option value="browser">Browser speech</option>
                    <option value="generated">Generated audio</option>
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
                      onChange={(event) => changeSpeechRate(Number(event.target.value))}
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
                    startPlaybackQueue(
                      wordPlaybackQueue(
                        result?.vocabulary_cards ?? [],
                        result?.target_language ?? targetLanguage,
                        playbackMode
                      ),
                      "Words"
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
                    startPlaybackQueue(
                      examplePlaybackQueue(
                        result?.vocabulary_cards ?? [],
                        result?.target_language ?? targetLanguage,
                        playbackMode
                      ),
                      "Examples"
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
                  aria-label="Previous item"
                  className="icon-button"
                  disabled={!speaking || playbackTotal < 2}
                  type="button"
                  onClick={() => skipPlayback(-1)}
                >
                  <SkipBack size={18} />
                </button>
                <button
                  aria-label="Next item"
                  className="icon-button"
                  disabled={!speaking || playbackTotal < 2}
                  type="button"
                  onClick={() => skipPlayback(1)}
                >
                  <SkipForward size={18} />
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
              <div className="playback-status" aria-live="polite">
                <span>{speaking ? playbackLabel || "Read Aloud" : "Background playback"}</span>
                <strong>
                  {audioLoading
                    ? "Generating..."
                    : speaking
                      ? `${playbackProgress(playbackCurrent, playbackTotal)} · ${
                          activeEngine === "generated" ? "audio" : "speech"
                        }`
                      : ttsConfig?.configured
                        ? `${ttsConfig.provider} ready`
                        : pwaReady
                          ? "PWA ready"
                          : "Browser speech"}
                </strong>
              </div>
              <p className="muted-copy">
                Browser speech is the default and does not call the generated-audio API. Select Generated audio only when
                you want provider audio for stronger mobile background playback; it may use paid API credits.
              </p>
              {!ttsConfig?.configured && playbackEngine !== "browser" ? (
                <p className="voice-warning">
                  Generated audio is not configured. Add OpenAI TTS settings on the API, or choose Browser speech.
                </p>
              ) : null}
              {ttsConfig?.configured && !user ? (
                <p className="voice-warning">Sign in to use generated audio. Browser speech still works without an account.</p>
              ) : null}
              {playbackNotice ? <p className="voice-warning">{playbackNotice}</p> : null}
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
