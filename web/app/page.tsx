"use client";

import {
  ArrowLeft,
  BookOpen,
  CheckCircle2,
  ChevronLeft,
  Compass,
  FileText,
  Flame,
  Gamepad2,
  Headphones,
  Heart,
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
  Trophy,
  Trash2,
  Upload,
  Volume2,
  XCircle
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  PlaybackEngine,
  PlaybackMode,
  PlaybackQueueItem,
  SAVED_LESSONS_KEY,
  SavedLesson,
  SavedLessonListResponse,
  SpeechSegment,
  StudyResponse,
  audioCacheKey,
  createSavedLesson,
  deserializeLessons,
  examplePlaybackQueue,
  exampleSegments,
  formatPlaybackRate,
  formatPair,
  MAX_PLAYBACK_RATE,
  MIN_PLAYBACK_RATE,
  normalizePlaybackRate,
  pageCount,
  paginateItems,
  PLAYBACK_RATE_PRESETS,
  PLAYBACK_RATE_STEP,
  playbackProgress,
  readAloudSource,
  serializeLessons,
  shouldUseGeneratedAudio,
  SpokenTextDisplay,
  spokenTextDisplay,
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
import {
  GameQuestion,
  PRACTICE_PROGRESS_KEY,
  PracticeChoice,
  PracticeMode,
  PracticeProgress,
  PracticeSession,
  answerMatches,
  applyPracticeAnswer,
  buildPracticeSession,
  deserializePracticeProgressMap,
  emptyPracticeProgress,
  practiceAudioSegments,
  progressAccuracy,
  serializePracticeProgressMap
} from "../lib/games";
import { SAMPLE_LESSONS } from "../lib/sampleLessons";

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
type AppScreen = "learn" | "results" | "practice" | "library" | "audio";
type ResultSection = "overview" | "vocabulary" | "textbook" | "units" | "note";
type LibrarySection = "lessons" | "guide" | "account";
type AudioSection = "playback" | "voice" | "speed" | "now";

type OAuthProvider = {
  provider: string;
  configured: boolean;
  authorization_url: string | null;
};

type OAuthRedeemResponse = {
  user: AuthUser;
  lessons: SavedLesson[];
  lessons_limit: number;
  lessons_offset: number;
  lessons_total: number;
  session_token: string;
};

type PracticeProgressResponse = {
  lesson_id: string;
  progress: PracticeProgress;
  exists: boolean;
  updated_at: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const ALLOW_REMEMBER_ME = process.env.NEXT_PUBLIC_ALLOW_REMEMBER_ME !== "false";
const LESSON_PAGE_LIMIT = 50;
const VOCAB_PAGE_SIZE = 8;
const TEXTBOOK_PAGE_SIZE = 1;
const UNIT_PAGE_SIZE = 8;
const levels = ["A1", "A2", "B1", "B2", "C1", "C2"];
const targetLanguages = ["English", "Turkish", "Spanish", "French", "German", "Italian"];
const practiceModes: Array<{ value: PracticeMode; label: string; description: string }> = [
  { value: "mix", label: "Recommended Mix", description: "A balanced arcade run from your uploaded words." },
  { value: "match", label: "Match Pairs", description: "Pair Turkish words with meanings." },
  { value: "listen", label: "Listen & Pick", description: "Hear Turkish and choose the translation." },
  { value: "recall", label: "Translation Recall", description: "See the meaning and recall Turkish." },
  { value: "sentence", label: "Sentence Builder", description: "Put example sentences in order." },
  { value: "blank", label: "Fill the Blank", description: "Complete a Turkish example." },
  { value: "chunk", label: "Chunk Builder", description: "Build phrases, verbs, and suffix-like chunks." }
];
const guideSections = [
  {
    title: "Understanding Turkish sentence structure",
    intro: "Read Turkish like a formula: find the verb, unpack the suffixes, then connect the details.",
    strategies: [
      {
        title: "Anchor and Reverse",
        purpose: "Use the final verb as the sentence anchor instead of translating from the first word.",
        drill: "Jump to the end, identify the action and tense, then move backward for who, what, where, and when.",
        example: "Arkadaşım dün kütüphanede çok zor bir kitap okudu.",
        hint: "okudu = read; then fill in friend, yesterday, library, difficult book."
      },
      {
        title: "Suffix Stack",
        purpose: "Long Turkish words become easier when you treat suffixes as ordered data.",
        drill: "Find the root first, then unpack the ending from right to left: person, tense, negation, voice.",
        example: "Yap-tır-ma-dı-m",
        hint: "I + past + not + make someone do + do = I did not have it done."
      },
      {
        title: "Russian Doll Clauses",
        purpose: "Relative clauses often sit before the noun as one large adjective.",
        drill: "When you see -en/-an or -dik/-dık forms, bundle the words before them and attach them to the next noun.",
        example: "[Dün aldığım] araba kırmızı.",
        hint: "The car that I bought yesterday is red."
      },
      {
        title: "Listening Buffer",
        purpose: "In conversation, hold early details until the final verb unlocks the meaning.",
        drill: "Listen for time, place, object, and case endings without translating; let the verb make them snap together.",
        example: "ev-e / ev-de",
        hint: "Tiny endings change the role: to the house vs. in the house."
      }
    ]
  },
  {
    title: "Speaking without translating from English",
    intro: "Build direct Turkish output so you are not composing a perfect English sentence first.",
    strategies: [
      {
        title: "Islands Technique",
        purpose: "Create small zones of fluent speech around topics you often discuss.",
        drill: "Prepare natural Turkish lines for work, hobbies, and recent events; repeat them until they feel automatic.",
        example: "Bugün yeni bir proje üzerinde çalışıyorum.",
        hint: "Pre-loaded sentences reduce pressure during real conversation."
      },
      {
        title: "Circumlocution",
        purpose: "Stay in Turkish when a word disappears by describing around it.",
        drill: "Pick an object and explain it without naming it, using only Turkish you already know.",
        example: "Bilgisayarıma elektrik veren kablo.",
        hint: "This keeps the Turkish circuit running instead of crashing back to English."
      },
      {
        title: "Lexical Chunking",
        purpose: "Learn usable blocks instead of isolated words.",
        drill: "Store collocations and verb phrases as one unit so the grammar is already baked in.",
        example: "hata yapmak",
        hint: "Learn make a mistake as one Turkish chunk, not two separate words."
      },
      {
        title: "Audio Shadowing",
        purpose: "Train your mouth and ear to operate before the translation layer starts.",
        drill: "Play a short Turkish clip and repeat out loud a fraction of a second behind the speaker without pausing.",
        example: "Dinle, takip et, aynı anda söyle.",
        hint: "No silence means less time for English word-order habits to interfere."
      }
    ]
  }
];

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

function lessonsPath(offset = 0, limit = LESSON_PAGE_LIMIT) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset)
  });
  return `/api/lessons?${params.toString()}`;
}

function isAuthRequired(error: unknown): boolean {
  return error instanceof Error && /authentication required|session expired/i.test(error.message);
}

function studyErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : "";
  if (/load failed|failed to fetch|networkerror/i.test(message)) {
    return "Could not reach the tutor API. Refresh and try again; if it persists, the backend service may be waking up or blocked.";
  }
  return message || "Study request failed.";
}

function GoogleIcon() {
  return (
    <svg aria-hidden="true" className="oauth-icon" viewBox="0 0 24 24">
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg aria-hidden="true" className="oauth-icon" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
    </svg>
  );
}

function oauthProviderLabel(provider: string): string {
  const normalized = provider.toLowerCase();
  if (normalized === "google") {
    return "Google";
  }
  if (normalized === "github") {
    return "GitHub";
  }
  return provider;
}

function oauthProviderIcon(provider: string) {
  const normalized = provider.toLowerCase();
  if (normalized === "google") {
    return <GoogleIcon />;
  }
  if (normalized === "github") {
    return <GitHubIcon />;
  }
  return null;
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
  const [showSpokenText, setShowSpokenText] = useState(false);
  const [currentSpokenText, setCurrentSpokenText] = useState<SpokenTextDisplay | null>(null);
  const [pwaReady, setPwaReady] = useState(false);
  const [playbackEngine, setPlaybackEngine] = useState<PlaybackEngine>("browser");
  const [appScreen, setAppScreen] = useState<AppScreen>("learn");
  const [resultSection, setResultSection] = useState<ResultSection>("overview");
  const [librarySection, setLibrarySection] = useState<LibrarySection>("lessons");
  const [audioSection, setAudioSection] = useState<AudioSection>("playback");
  const [vocabPage, setVocabPage] = useState(1);
  const [textbookPage, setTextbookPage] = useState(1);
  const [unitPage, setUnitPage] = useState(1);
  const [guidePage, setGuidePage] = useState(1);
  const [, setWorkspaceTab] = useState<"account" | "lessons" | "practice" | "guide" | "audio">("account");
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
  const [lessonPage, setLessonPage] = useState({ limit: LESSON_PAGE_LIMIT, offset: 0, total: 0 });
  const [lessonsLoaded, setLessonsLoaded] = useState(false);
  const [lessonsLoading, setLessonsLoading] = useState(false);
  const [lessonError, setLessonError] = useState("");
  const [lessonTitle, setLessonTitle] = useState("");
  const [lessonSearch, setLessonSearch] = useState("");
  const [activeLessonId, setActiveLessonId] = useState<string | null>(null);
  const [practiceMode, setPracticeMode] = useState<PracticeMode>("mix");
  const [practiceSession, setPracticeSession] = useState<PracticeSession | null>(null);
  const [practiceIndex, setPracticeIndex] = useState(0);
  const [practiceProgress, setPracticeProgress] = useState<PracticeProgress>(() => emptyPracticeProgress());
  const [practiceLoading, setPracticeLoading] = useState(false);
  const [practiceError, setPracticeError] = useState("");
  const [practiceFeedback, setPracticeFeedback] = useState<{ correct: boolean; message: string } | null>(null);
  const [practiceHearts, setPracticeHearts] = useState(5);
  const [practiceStreak, setPracticeStreak] = useState(0);
  const [practiceMistakes, setPracticeMistakes] = useState(0);
  const [practiceRoundCorrect, setPracticeRoundCorrect] = useState(0);
  const [practiceFirstTry, setPracticeFirstTry] = useState(true);
  const [practiceTypedAnswer, setPracticeTypedAnswer] = useState("");
  const [practiceBuiltParts, setPracticeBuiltParts] = useState<PracticeChoice[]>([]);
  const [practiceMatchedIds, setPracticeMatchedIds] = useState<string[]>([]);
  const [practiceSelectedMatch, setPracticeSelectedMatch] = useState<{ side: "turkish" | "translation"; id: string } | null>(null);
  const [localPracticeProgress, setLocalPracticeProgress] = useState<Record<string, PracticeProgress>>(() => {
    if (typeof window === "undefined") {
      return {};
    }
    return deserializePracticeProgressMap(window.localStorage.getItem(PRACTICE_PROGRESS_KEY));
  });
  const playbackQueueRef = useRef<PlaybackQueueItem[]>([]);
  const playbackItemIndexRef = useRef(0);
  const playbackSegmentIndexRef = useRef(0);
  const playbackRunRef = useRef(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCacheRef = useRef<Map<string, string>>(new Map());
  const speechRateRef = useRef(1);
  const restartBrowserOnResumeRef = useRef(false);
  const browserSpeechWatchdogRef = useRef<number | null>(null);
  const pausedRef = useRef(false);
  const tabPanelRef = useRef<HTMLElement | null>(null);
  const practiceGameRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  const currentPracticeKey = useMemo(() => {
    if (activeLessonId) {
      return `lesson:${activeLessonId}`;
    }
    if (!result) {
      return "draft:empty";
    }
    return `draft:${result.source_type}:${result.source_label}:${result.study_level}:${result.preview.slice(0, 80)}`;
  }, [activeLessonId, result]);

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
    setLessonPage({ limit: LESSON_PAGE_LIMIT, offset: 0, total: local.length });
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
            : await (async () => {
                const auth = await apiJson<AuthResponse>("/api/auth/me");
                const lessonPage = await apiJson<SavedLessonListResponse>(lessonsPath());
                return {
                  user: auth.user,
                  lessons: lessonPage.lessons,
                  lessons_limit: lessonPage.limit,
                  lessons_offset: lessonPage.offset,
                  lessons_total: lessonPage.total,
                  session_token: null
                };
              })();
          storeSessionToken(payload.session_token, rememberOAuth);
          setUser(payload.user);
          setSavedLessons(payload.lessons);
          setLessonPage({
            limit: payload.lessons_limit,
            offset: payload.lessons_offset,
            total: payload.lessons_total
          });
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
    window.localStorage.setItem(PRACTICE_PROGRESS_KEY, serializePracticeProgressMap(localPracticeProgress));
  }, [localPracticeProgress]);

  useEffect(() => {
    let cancelled = false;
    if (!result) {
      setPracticeProgress(emptyPracticeProgress());
      setPracticeSession(null);
      setPracticeError("");
      return;
    }

    if (user && activeLessonId) {
      setPracticeLoading(true);
      setPracticeError("");
      void apiJson<PracticeProgressResponse>(`/api/practice/progress?lesson_id=${encodeURIComponent(activeLessonId)}`)
        .then((payload) => {
          if (!cancelled) {
            setPracticeProgress(payload.progress ? payload.progress : emptyPracticeProgress());
          }
        })
        .catch((caught) => {
          if (!cancelled) {
            setPracticeProgress(emptyPracticeProgress());
            setPracticeError(caught instanceof Error ? caught.message : "Could not load practice progress.");
          }
        })
        .finally(() => {
          if (!cancelled) {
            setPracticeLoading(false);
          }
        });
      return () => {
        cancelled = true;
      };
    }

    setPracticeProgress(localPracticeProgress[currentPracticeKey] ?? emptyPracticeProgress());
    setPracticeError("");
    return () => {
      cancelled = true;
    };
  }, [result, user, activeLessonId, localPracticeProgress, currentPracticeKey]);

  useEffect(() => {
    if (!("serviceWorker" in navigator)) {
      return;
    }
    const canRegister = window.location.protocol === "https:" || window.location.hostname === "localhost";
    if (!canRegister) {
      return;
    }
    const hadServiceWorkerController = Boolean(navigator.serviceWorker.controller);
    let reloadingForServiceWorker = false;
    const handleControllerChange = () => {
      if (!hadServiceWorkerController) {
        return;
      }
      if (reloadingForServiceWorker) {
        return;
      }
      reloadingForServiceWorker = true;
      window.location.reload();
    };
    navigator.serviceWorker.addEventListener("controllerchange", handleControllerChange);
    navigator.serviceWorker
      .register("/sw.js", { updateViaCache: "none" })
      .then((registration) => {
        setPwaReady(true);
        if (registration.waiting) {
          registration.waiting.postMessage({ type: "SKIP_WAITING" });
        }
        void registration.update();
      })
      .catch(() => setPwaReady(false));
    return () => {
      navigator.serviceWorker.removeEventListener("controllerchange", handleControllerChange);
    };
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
          const lessonPage = await apiJson<SavedLessonListResponse>(lessonsPath());
          setSavedLessons(lessonPage.lessons);
          setLessonPage({ limit: lessonPage.limit, offset: lessonPage.offset, total: lessonPage.total });
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

  const readableSource = useMemo(() => readAloudSource(result, text), [result, text]);

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
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || payload.message || `Study request failed (${response.status}).`);
      }
      setResult(payload);
      setLevel(payload.study_level);
      setActiveLessonId(null);
      setLessonTitle(defaultLessonTitle(payload));
      setSearch("");
      setTypeFilter("all");
      setResultSection("overview");
      setVocabPage(1);
      setTextbookPage(1);
      setUnitPage(1);
      setAppScreen("results");
      resetPracticeSession();
    } catch (caught) {
      setError(studyErrorMessage(caught));
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
          title: item.title || "Türkçe Hoca",
          artist: item.subtitle || "Turkish tutor",
          album: playbackLabel || "Read Aloud"
        })
      : null;
  }

  function clearBrowserSpeechWatchdog() {
    if (browserSpeechWatchdogRef.current !== null) {
      window.clearInterval(browserSpeechWatchdogRef.current);
      browserSpeechWatchdogRef.current = null;
    }
  }

  function speechSegmentTimeoutMs(text: string) {
    const words = Math.max(text.split(/\s+/).filter(Boolean).length, 1);
    const rate = Math.max(speechRateRef.current, 0.25);
    return Math.max(14000, Math.min(90000, (words / rate) * 750 + 10000));
  }

  function resetPlaybackState(message = "") {
    clearBrowserSpeechWatchdog();
    audioRef.current?.pause();
    audioRef.current = null;
    restartBrowserOnResumeRef.current = false;
    pausedRef.current = false;
    setSpeaking(false);
    setPaused(false);
    setAudioLoading(false);
    setActiveEngine("idle");
    setPlaybackCurrent(0);
    setPlaybackTotal(0);
    setPlaybackLabel("");
    setPlaybackNotice(message);
    setCurrentSpokenText(null);
    setMediaSession(null, "none");
  }

  function speakSegment(segment: SpeechSegment, runId: number) {
    clearBrowserSpeechWatchdog();
    setAudioLoading(false);
    setActiveEngine("browser");
    const utterance = new SpeechSynthesisUtterance(segment.text);
    let settled = false;
    const startedAt = Date.now();
    const timeoutMs = speechSegmentTimeoutMs(segment.text);
    utterance.lang = segment.lang;
    utterance.rate = speechRateRef.current;
    const voice = selectVoiceForLanguage(segment.lang);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
    utterance.onend = () => {
      if (settled || runId !== playbackRunRef.current) {
        return;
      }
      settled = true;
      clearBrowserSpeechWatchdog();
      playbackSegmentIndexRef.current += 1;
      void playCurrentSegment();
    };
    utterance.onerror = () => {
      if (settled || runId !== playbackRunRef.current) {
        return;
      }
      settled = true;
      clearBrowserSpeechWatchdog();
      resetPlaybackState("Playback stopped. This browser may have blocked background speech.");
    };
    browserSpeechWatchdogRef.current = window.setInterval(() => {
      if (settled || runId !== playbackRunRef.current) {
        clearBrowserSpeechWatchdog();
        return;
      }
      if (pausedRef.current) {
        return;
      }
      window.speechSynthesis.resume();
      if (Date.now() - startedAt > timeoutMs) {
        settled = true;
        clearBrowserSpeechWatchdog();
        window.speechSynthesis.cancel();
        playbackSegmentIndexRef.current += 1;
        setPlaybackNotice("Browser speech stalled, so Türkçe Hoca moved to the next line automatically.");
        void playCurrentSegment();
      }
    }, 8000);
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
    restartBrowserOnResumeRef.current = false;
    pausedRef.current = false;
    setSpeaking(true);
    setPaused(false);
    setPlaybackCurrent(playbackItemIndexRef.current);
    setPlaybackTotal(queue.length);
    setCurrentSpokenText(
      spokenTextDisplay(item, segment, playbackItemIndexRef.current, queue.length, playbackSegmentIndexRef.current)
    );
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
    clearBrowserSpeechWatchdog();
    window.speechSynthesis?.cancel();
    audioRef.current?.pause();
    audioRef.current = null;
    restartBrowserOnResumeRef.current = false;
    playbackQueueRef.current = queue;
    playbackItemIndexRef.current = 0;
    playbackSegmentIndexRef.current = 0;
    setPlaybackLabel(label);
    setPlaybackTotal(queue.length);
    setPlaybackCurrent(0);
    setCurrentSpokenText(null);
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
          subtitle: "Türkçe Hoca",
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
    speakTexts([readableSource.text]);
  }

  function resetPracticeInteraction() {
    setPracticeFeedback(null);
    setPracticeFirstTry(true);
    setPracticeTypedAnswer("");
    setPracticeBuiltParts([]);
    setPracticeMatchedIds([]);
    setPracticeSelectedMatch(null);
  }

  function resetPracticeSession() {
    setPracticeSession(null);
    setPracticeIndex(0);
    setPracticeHearts(5);
    setPracticeStreak(0);
    setPracticeMistakes(0);
    setPracticeRoundCorrect(0);
    setPracticeError("");
    resetPracticeInteraction();
  }

  function switchWorkspaceTab(nextTab: "account" | "lessons" | "practice" | "guide" | "audio") {
    setWorkspaceTab(nextTab);
    if (nextTab === "practice") {
      setAppScreen("practice");
    } else if (nextTab === "audio") {
      setAppScreen("audio");
    } else {
      setAppScreen("library");
      setLibrarySection(nextTab === "guide" ? "guide" : nextTab === "account" ? "account" : "lessons");
    }
    window.setTimeout(() => {
      tabPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 30);
  }

  async function persistPracticeProgress(next: PracticeProgress) {
    setPracticeProgress(next);
    if (user && activeLessonId) {
      try {
        const saved = await apiJson<PracticeProgressResponse>("/api/practice/progress", {
          method: "PUT",
          body: JSON.stringify({ lesson_id: activeLessonId, progress: next })
        });
        setPracticeProgress(saved.progress ?? next);
      } catch (caught) {
        setPracticeError(caught instanceof Error ? caught.message : "Could not save practice progress.");
      }
      return;
    }
    setLocalPracticeProgress((current) => ({ ...current, [currentPracticeKey]: next }));
  }

  async function syncPracticeProgressForLesson(lessonId: string) {
    if (!user || practiceProgress.attempts === 0) {
      return;
    }
    try {
      await apiJson<PracticeProgressResponse>("/api/practice/progress", {
        method: "PUT",
        body: JSON.stringify({ lesson_id: lessonId, progress: practiceProgress })
      });
    } catch {
      // Saving the lesson itself is more important; the next game answer will retry progress sync.
    }
  }

  function startPractice(nextMode = practiceMode) {
    if (!result?.vocabulary_cards.length) {
      setPracticeError("Analyze or open a lesson with vocabulary cards before starting practice.");
      return;
    }
    const session = buildPracticeSession(result, {
      mode: nextMode,
      seed: `${currentPracticeKey}:${practiceProgress.attempts}:${nextMode}`,
      progress: practiceProgress
    });
    if (!session.questions.length) {
      setPracticeError("This lesson needs at least two usable vocabulary cards for games.");
      return;
    }
    stopSpeech();
    setPracticeMode(nextMode);
    setPracticeSession(session);
    setPracticeIndex(0);
    setPracticeHearts(5);
    setPracticeStreak(0);
    setPracticeMistakes(0);
    setPracticeRoundCorrect(0);
    setPracticeError("");
    resetPracticeInteraction();
    switchWorkspaceTab("practice");
    setAppScreen("practice");
  }

  function finishPracticeQuestion(question: GameQuestion, correct: boolean) {
    const nextProgress = applyPracticeAnswer(practiceProgress, question, correct, practiceFirstTry);
    void persistPracticeProgress(nextProgress);
    if (correct) {
      setPracticeRoundCorrect((current) => current + 1);
      setPracticeStreak((current) => current + 1);
      setPracticeFeedback({ correct: true, message: practiceFirstTry ? "Clean hit." : "Correct on review." });
      return;
    }
    setPracticeMistakes((current) => current + 1);
    setPracticeStreak(0);
    setPracticeFirstTry(false);
    setPracticeHearts((current) => Math.max(0, current - 1));
    setPracticeFeedback({ correct: false, message: "Missed. Review the answer, then continue." });
  }

  function answerPracticeChoice(answer: string) {
    const question = currentPracticeQuestion;
    if (!question || practiceFeedback) {
      return;
    }
    finishPracticeQuestion(question, answerMatches(question, answer));
  }

  function submitBuiltPracticeAnswer() {
    const question = currentPracticeQuestion;
    if (!question || practiceFeedback || !practiceBuiltParts.length) {
      return;
    }
    finishPracticeQuestion(question, answerMatches(question, practiceBuiltParts.map((part) => part.text)));
  }

  function submitTypedPracticeAnswer() {
    const question = currentPracticeQuestion;
    if (!question || practiceFeedback || !practiceTypedAnswer.trim()) {
      return;
    }
    finishPracticeQuestion(question, answerMatches(question, practiceTypedAnswer));
  }

  function choosePracticePart(choice: PracticeChoice) {
    if (practiceFeedback || practiceBuiltParts.some((part) => part.id === choice.id)) {
      return;
    }
    setPracticeBuiltParts((current) => [...current, choice]);
  }

  function removePracticePart(choice: PracticeChoice) {
    if (practiceFeedback) {
      return;
    }
    setPracticeBuiltParts((current) => current.filter((part) => part.id !== choice.id));
  }

  function choosePracticeMatch(side: "turkish" | "translation", id: string) {
    const question = currentPracticeQuestion;
    if (!question || question.activity !== "match" || practiceFeedback || practiceMatchedIds.includes(id)) {
      return;
    }
    if (!practiceSelectedMatch || practiceSelectedMatch.side === side) {
      setPracticeSelectedMatch({ side, id });
      return;
    }
    if (practiceSelectedMatch.id === id) {
      const nextMatched = [...practiceMatchedIds, id];
      setPracticeMatchedIds(nextMatched);
      setPracticeSelectedMatch(null);
      if (nextMatched.length === question.matchPairs.length) {
        finishPracticeQuestion(question, true);
      }
      return;
    }
    setPracticeSelectedMatch(null);
    setPracticeFirstTry(false);
    setPracticeMistakes((current) => current + 1);
    setPracticeStreak(0);
    setPracticeHearts((current) => Math.max(0, current - 1));
    setPracticeFeedback({ correct: false, message: "Those two do not match. Try another pair." });
    window.setTimeout(() => setPracticeFeedback(null), 900);
  }

  function nextPracticeQuestion() {
    if (!practiceSession) {
      return;
    }
    const nextIndex = practiceIndex + 1;
    resetPracticeInteraction();
    setPracticeIndex(nextIndex);
  }

  function retryPracticeMisses() {
    if (!result) {
      return;
    }
    startPractice(practiceProgress.missedCardIds.length ? "boss" : "mix");
  }

  function speakPracticeQuestion(question: GameQuestion) {
    const segments = practiceAudioSegments(question, result?.target_language || targetLanguage);
    if (!segments.length) {
      setPlaybackNotice("No practice audio is available for this question.");
      return;
    }
    speakSegments(segments, question.title);
  }

  function applyLessonPage(payload: SavedLessonListResponse, append = false) {
    setSavedLessons((current) => (append ? [...current, ...payload.lessons.filter((lesson) => !current.some((item) => item.id === lesson.id))] : payload.lessons));
    setLessonPage({ limit: payload.limit, offset: payload.offset, total: payload.total });
  }

  async function loadRemoteLessons(offset = 0, append = false) {
    setLessonsLoading(true);
    setLessonError("");
    try {
      const lessonPage = await apiJson<SavedLessonListResponse>(lessonsPath(offset));
      applyLessonPage(lessonPage, append);
    } catch (caught) {
      if (isAuthRequired(caught)) {
        try {
          await apiJson<AuthResponse>("/api/auth/me");
          const lessonPage = await apiJson<SavedLessonListResponse>(lessonsPath(offset));
          applyLessonPage(lessonPage, append);
          return;
        } catch {
          setUser(null);
          setSavedLessons(localLessons);
          setLessonPage({ limit: LESSON_PAGE_LIMIT, offset: 0, total: localLessons.length });
          setAuthError("Your sign-in session was not accepted. Please log in again.");
          return;
        }
      }
      setLessonError(caught instanceof Error ? caught.message : "Could not load saved lessons.");
    } finally {
      setLessonsLoading(false);
    }
  }

  function loadMoreLessons() {
    void loadRemoteLessons(savedLessons.length, true);
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
      setLessonPage({ limit: LESSON_PAGE_LIMIT, offset: 0, total: localLessons.length });
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
        if (!lesson.result) {
          continue;
        }
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
        if (!existing) {
          setLessonPage((current) => ({ ...current, total: current.total + 1 }));
        }
        setActiveLessonId(saved.id);
        setLessonTitle(saved.title);
        await syncPracticeProgressForLesson(saved.id);
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
      setLessonPage((current) => ({ ...current, total: upsertLesson(savedLessons, lesson).length }));
      setActiveLessonId(lesson.id);
      setLessonTitle(lesson.title);
    } catch (caught) {
      setLessonError(caught instanceof Error ? caught.message : "Could not save lesson.");
    }
  }

  async function openLesson(lesson: SavedLesson, targetScreen: "results" | "practice" = "results") {
    stopSpeech();
    setLessonError("");
    let fullLesson = lesson;
    if (!fullLesson.result && user) {
      setLessonsLoading(true);
      try {
        fullLesson = await apiJson<SavedLesson>(`/api/lessons/${lesson.id}`);
        setSavedLessons((current) => upsertLesson(current, fullLesson));
      } catch (caught) {
        setLessonError(caught instanceof Error ? caught.message : "Could not open saved lesson.");
        return;
      } finally {
        setLessonsLoading(false);
      }
    }
    if (!fullLesson.result) {
      setLessonError("This saved lesson summary is missing its full study result. Refresh and try again.");
      return;
    }
    setResult(fullLesson.result);
    setLevel(fullLesson.result.study_level);
    setTargetLanguage(fullLesson.result.target_language);
    setSearch("");
    setTypeFilter("all");
    setResultSection("overview");
    setVocabPage(1);
    setTextbookPage(1);
    setUnitPage(1);
    setAppScreen(targetScreen);
    setActiveLessonId(fullLesson.id);
    setLessonTitle(fullLesson.title);
    setError("");
    resetPracticeSession();
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
      setLessonPage((current) => ({ ...current, total: Math.max(0, current.total - 1) }));
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
        if (restartBrowserOnResumeRef.current) {
          restartBrowserOnResumeRef.current = false;
          void playCurrentSegment();
        } else {
          window.speechSynthesis.resume();
        }
      }
      setPaused(false);
      if ("mediaSession" in navigator) {
        navigator.mediaSession.playbackState = "playing";
      }
      pausedRef.current = false;
    } else {
      if (activeEngine === "generated") {
        audioRef.current?.pause();
      } else if ("speechSynthesis" in window) {
        window.speechSynthesis.pause();
      }
      setPaused(true);
      pausedRef.current = true;
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
    restartBrowserOnResumeRef.current = false;
    clearBrowserSpeechWatchdog();
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
      playbackQueueRef.current.length &&
      "speechSynthesis" in window
    ) {
      playbackRunRef.current += 1;
      clearBrowserSpeechWatchdog();
      window.speechSynthesis.cancel();
      if (paused) {
        restartBrowserOnResumeRef.current = true;
        setPlaybackNotice(`Speed set to ${formatPlaybackRate(normalized)}. Resume restarts the current line at the new speed.`);
      } else {
        void playCurrentSegment();
      }
    }
  }

  function stopSpeech() {
    clearBrowserSpeechWatchdog();
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

  function lessonMeta(lesson: SavedLesson) {
    if (lesson.id.startsWith("sample-")) {
      return `${lesson.result?.study_level || "Any"} · Default Curated Resource`;
    }
    if (!lesson.result) {
      return `Saved · ${new Date(lesson.created_at).toLocaleDateString()}`;
    }
    return `${lesson.result.study_level} · ${lesson.result.target_language} · ${new Date(lesson.created_at).toLocaleDateString()}`;
  }

  const filteredLessons = useMemo(() => {
    const query = lessonSearch.trim().toLowerCase();
    if (!query) {
      return savedLessons;
    }
    return savedLessons.filter((lesson) =>
      [
        lesson.title,
        lesson.result?.source_label,
        lesson.result?.preview,
        lesson.result?.target_language,
        lesson.result?.study_level
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  }, [savedLessons, lessonSearch]);

  const filteredSampleLessons = useMemo(() => {
    const query = lessonSearch.trim().toLowerCase();
    if (!query) {
      return SAMPLE_LESSONS;
    }
    return SAMPLE_LESSONS.filter((lesson) =>
      [
        lesson.title,
        lesson.result?.source_label,
        lesson.result?.preview,
        lesson.result?.target_language,
        lesson.result?.study_level
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  }, [lessonSearch]);

  const pagedCards = useMemo(() => paginateItems(filteredCards, vocabPage, VOCAB_PAGE_SIZE), [filteredCards, vocabPage]);
  const pagedTextbookSections = useMemo(
    () => paginateItems(result?.textbook_sections ?? [], textbookPage, TEXTBOOK_PAGE_SIZE),
    [result, textbookPage]
  );
  const pagedUnits = useMemo(() => paginateItems(result?.units ?? [], unitPage, UNIT_PAGE_SIZE), [result, unitPage]);
  const pagedGuideSections = useMemo(() => paginateItems(guideSections, guidePage, 1), [guidePage]);
  const hasResult = Boolean(result);
  const resultTabs = useMemo(
    () =>
      [
        { value: "overview" as const, label: "Overview", count: result ? 1 : 0 },
        { value: "vocabulary" as const, label: "Vocabulary", count: result?.vocabulary_cards.length ?? 0 },
        { value: "textbook" as const, label: "Textbook", count: result?.textbook_sections?.length ?? 0 },
        { value: "units" as const, label: "Units", count: result?.units.length ?? 0 },
        { value: "note" as const, label: "Note", count: result?.note ? 1 : 0 }
      ],
    [result]
  );
  const currentGuideSection = pagedGuideSections.items[0];

  useEffect(() => {
    setVocabPage((current) => Math.min(current, pageCount(filteredCards.length, VOCAB_PAGE_SIZE)));
  }, [filteredCards.length]);

  useEffect(() => {
    setTextbookPage((current) => Math.min(current, pageCount(result?.textbook_sections?.length ?? 0, TEXTBOOK_PAGE_SIZE)));
    setUnitPage((current) => Math.min(current, pageCount(result?.units.length ?? 0, UNIT_PAGE_SIZE)));
  }, [result]);

  const currentPracticeQuestion = practiceSession?.questions[practiceIndex] ?? null;
  const activePracticeSessionId = practiceSession?.id ?? "";
  const practiceComplete = Boolean(practiceSession && (practiceIndex >= practiceSession.questions.length || practiceHearts <= 0));
  const practiceProgressPercent = practiceSession?.questions.length
    ? Math.min(100, Math.round((Math.min(practiceIndex, practiceSession.questions.length) / practiceSession.questions.length) * 100))
    : 0;
  const practiceCanContinue = Boolean(practiceFeedback && (practiceFeedback.correct || currentPracticeQuestion?.activity !== "match" || practiceHearts <= 0));

  useEffect(() => {
    if (appScreen !== "practice" || !practiceSession || practiceComplete) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      const question = currentPracticeQuestion;
      if (!question) {
        return;
      }
      if (event.key === "Enter") {
        if (practiceCanContinue) {
          event.preventDefault();
          nextPracticeQuestion();
          return;
        }
        if (question.activity === "sentence" || question.activity === "chunk") {
          event.preventDefault();
          submitBuiltPracticeAnswer();
          return;
        }
        if (question.activity === "recall" && practiceTypedAnswer.trim()) {
          event.preventDefault();
          submitTypedPracticeAnswer();
        }
      }
      const numeric = Number(event.key);
      if (Number.isInteger(numeric) && numeric >= 1 && numeric <= question.choices.length) {
        event.preventDefault();
        const choice = question.choices[numeric - 1];
        if (question.activity === "sentence" || question.activity === "chunk") {
          choosePracticePart(choice);
        } else {
          answerPracticeChoice(choice.text);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
    // Keyboard shortcuts delegate to the current render's practice handlers; keeping the list focused avoids callback ceremony here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    appScreen,
    practiceSession,
    practiceComplete,
    currentPracticeQuestion,
    practiceCanContinue,
    practiceTypedAnswer,
    practiceBuiltParts,
    practiceFeedback
  ]);

  useEffect(() => {
    if (appScreen !== "practice" || !activePracticeSessionId) {
      return;
    }
    const timeout = window.setTimeout(() => {
      practiceGameRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      practiceGameRef.current?.focus({ preventScroll: true });
    }, 40);
    return () => window.clearTimeout(timeout);
  }, [appScreen, activePracticeSessionId]);

  const activeLesson = activeLessonId
    ? savedLessons.find((lesson) => lesson.id === activeLessonId)
    : null;

  function openAppScreen(nextScreen: AppScreen) {
    setAppScreen(nextScreen);
    if (nextScreen === "practice") {
      setWorkspaceTab("practice");
    }
    if (nextScreen === "audio") {
      setWorkspaceTab("audio");
    }
    window.setTimeout(() => {
      tabPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 30);
  }

  function pageControls(label: string, page: number, totalPages: number, onPage: (nextPage: number) => void) {
    return (
      <div className="page-controls" aria-label={`${label} pagination`}>
        <button className="ghost-button" disabled={page <= 1} type="button" onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <span>
          {label}: {page} / {totalPages}
        </span>
        <button className="ghost-button" disabled={page >= totalPages} type="button" onClick={() => onPage(page + 1)}>
          Next
        </button>
      </div>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">TR</div>
          <div>
            <h1>Türkçe Hoca</h1>
            <p>CEFR-aware Turkish tutor workspace</p>
          </div>
        </div>
        <div className="status">
          <span className={`status-dot ${health?.gemini_ready ? "ready" : ""}`} />
          <span>
            {health?.gemini_ready ? "Gemini ready" : "API waiting"} · {health?.topics ?? 0} topics
          </span>
        </div>
        <button className="account-shortcut" type="button" onClick={() => { setLibrarySection("account"); openAppScreen("library"); }}>
          {user ? <KeyRound size={16} /> : <LogIn size={16} />}
          <span>{user ? "Account" : "Sign in"}</span>
        </button>
      </header>

      <nav className="app-nav" aria-label="Primary app screens">
        <button className={appScreen === "learn" ? "active" : ""} type="button" onClick={() => openAppScreen("learn")}>
          <FileText size={18} />
          <span>Learn</span>
        </button>
        <button className={appScreen === "results" ? "active" : ""} type="button" onClick={() => openAppScreen("results")}>
          <BookOpen size={18} />
          <span>Results</span>
        </button>
        <button className={appScreen === "practice" ? "active" : ""} type="button" onClick={() => openAppScreen("practice")}>
          <Gamepad2 size={18} />
          <span>Practice</span>
        </button>
        <button className={appScreen === "library" ? "active" : ""} type="button" onClick={() => openAppScreen("library")}>
          <Compass size={18} />
          <span>Library</span>
        </button>
        <button className={appScreen === "audio" ? "active" : ""} type="button" onClick={() => openAppScreen("audio")}>
          <Headphones size={18} />
          <span>Audio</span>
        </button>
      </nav>

      <section className={`app-screen app-screen-${appScreen}`} ref={tabPanelRef}>
        {appScreen === "learn" ? (
          <div className="screen-grid learn-grid">
            <form className="panel input-panel" onSubmit={submitStudy}>
          <div className="panel-header">
            <div className="panel-title">
              <FileText size={18} />
                  <h2>Learn</h2>
            </div>
          </div>
          <div className="panel-body">
                <div className="screen-intro">
                  <span className="pill">Step 1</span>
                  <h2>Upload, paste, or type Turkish.</h2>
                  <p>Türkçe Hoca turns your material into vocabulary cards, textbook notes, games, and read-aloud practice.</p>
                </div>
            <div className="field">
              <label htmlFor="text-input">Text</label>
              <textarea id="text-input" value={text} onChange={(event) => setText(event.target.value)} />
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
                  <button className={item === level ? "active" : ""} key={item} type="button" onClick={() => setLevel(item)}>
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <label htmlFor="target-language">Target</label>
              <select id="target-language" value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)}>
                {targetLanguages.map((language) => (
                  <option key={language}>{language}</option>
                ))}
              </select>
            </div>

            <button className="primary-button" disabled={loading} type="submit">
              {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              Analyze
            </button>

            <p className="muted-copy">
              Privacy: text and uploaded files are sent to Gemini for analysis. Generated audio sends selected text only
              when you choose Generated audio.
            </p>
            {error ? <div className="error">{error}</div> : null}
          </div>
        </form>

            <section className="panel flow-panel">
              <div className="panel-header">
                <div className="panel-title">
                  <Gamepad2 size={18} />
                  <h2>Next steps</h2>
                </div>
              </div>
              <div className="panel-body">
                <div className="flow-actions">
                  <button className="ghost-button" disabled={!hasResult} type="button" onClick={() => openAppScreen("results")}>
                    <BookOpen size={18} />
                    View results
                  </button>
                  <button className="ghost-button" disabled={!result?.vocabulary_cards.length} type="button" onClick={() => startPractice(practiceMode)}>
                    <Gamepad2 size={18} />
                    Start practice
                  </button>
                  <button className="ghost-button" type="button" onClick={() => openAppScreen("library")}>
                    <Compass size={18} />
                    Open library
                  </button>
                  <button className="ghost-button" type="button" onClick={() => openAppScreen("audio")}>
                    <Headphones size={18} />
                    Read aloud
                  </button>
                </div>
                {result ? (
                  <div className="mini-summary">
                    <span className="pill">{result.study_level}</span>
                    <strong>{result.vocabulary_cards.length} vocabulary cards ready</strong>
                    <p>{result.preview.slice(0, 180)}</p>
                  </div>
                ) : (
                  <p className="muted-copy">Analyze something first, then the app will unlock Results and Practice.</p>
                )}
              </div>
            </section>
          </div>
        ) : null}

        {appScreen === "library" ? (
          <section className="panel tab-panel">
            <div className="section-tabs" role="tablist" aria-label="Library sections">
              <button className={librarySection === "lessons" ? "active" : ""} type="button" onClick={() => setLibrarySection("lessons")}>
                <BookOpen size={16} />
                Lessons
              </button>
              <button className={librarySection === "guide" ? "active" : ""} type="button" onClick={() => setLibrarySection("guide")}>
                <Compass size={16} />
                Guide
              </button>
              <button className={librarySection === "account" ? "active" : ""} type="button" onClick={() => setLibrarySection("account")}>
                <KeyRound size={16} />
                Account
              </button>
            </div>
            <div className="panel-body tab-body">
              {librarySection === "account" ? (
                <>
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
                      <button className={authMode === "login" ? "active" : ""} type="button" onClick={() => setAuthMode("login")}>
                        Login
                      </button>
                      <button className={authMode === "signup" ? "active" : ""} type="button" onClick={() => setAuthMode("signup")}>
                        Sign up
                      </button>
                      <button className={authMode === "reset" ? "active" : ""} type="button" onClick={() => setAuthMode("reset")}>
                        Reset
                      </button>
                    </div>

                    {authMode === "signup" ? (
                      <div className="field compact-field">
                        <label htmlFor="auth-name">Name</label>
                        <input id="auth-name" type="text" autoComplete="name" value={authName} onChange={(event) => setAuthName(event.target.value)} />
                      </div>
                    ) : null}

                    <div className="auth-grid">
                      <div className="field compact-field">
                        <label htmlFor="auth-email">Email</label>
                        <input id="auth-email" type="email" autoComplete="email" value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} />
                      </div>
                      <div className="field compact-field">
                        <label htmlFor="auth-password">{authMode === "reset" && resetToken ? "New password" : "Password"}</label>
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
                        <input checked={rememberMe} id="remember-me" type="checkbox" onChange={(event) => setRememberMe(event.target.checked)} />
                        <span>
                          <strong>Remember me</strong>
                          <small>Only use this on your own device.</small>
                        </span>
                      </label>
                    ) : null}

                    <button className="primary-button" disabled={authLoading} type="submit">
                      {authLoading ? <Loader2 className="spin" size={18} /> : <KeyRound size={18} />}
                      {authMode === "signup" ? "Create account" : authMode === "reset" && !resetToken ? "Request reset" : "Continue"}
                    </button>
                  </form>
                )}

                {!user && oauthProviders.length ? (
                  <div className="oauth-row">
                    {oauthProviders.map((provider) => (
                      <button
                        className="ghost-button oauth-button"
                        disabled={!provider.authorization_url}
                        key={provider.provider}
                        type="button"
                        onClick={() => startOAuth(provider)}
                        title={
                          provider.authorization_url
                            ? `Continue with ${oauthProviderLabel(provider.provider)}.`
                            : provider.configured
                              ? `${oauthProviderLabel(provider.provider)} credentials are configured; callback routes still need to be wired.`
                              : `${oauthProviderLabel(provider.provider)} OAuth is not configured yet.`
                        }
                      >
                        {oauthProviderIcon(provider.provider)}
                        <span className="oauth-label">{oauthProviderLabel(provider.provider)} OAuth</span>
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
              </>
            ) : null}

              {librarySection === "lessons" ? (
              <>
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
                {!user ? <p className="muted-copy">Not signed in: lessons are saved only in this browser until you log in.</p> : null}
                <div className="search-box lesson-search">
                  <Search size={16} />
                  <input
                    aria-label="Search saved lessons"
                    placeholder="Search saved lessons"
                    value={lessonSearch}
                    onChange={(event) => setLessonSearch(event.target.value)}
                  />
                  {lessonSearch ? (
                    <button className="clear-search-button" type="button" aria-label="Clear search" onClick={() => setLessonSearch("")}>
                      <XCircle size={16} />
                    </button>
                  ) : null}
                </div>
                {filteredLessons.length ? (
                  <div className="lesson-list">
                    {filteredLessons.map((lesson) => (
                      <article className={`lesson-card ${lesson.id === activeLessonId ? "active" : ""}`} key={lesson.id}>
                        <button className="lesson-open" type="button" onClick={() => void openLesson(lesson)}>
                          <strong>{lesson.title}</strong>
                          <span>{lessonMeta(lesson)}</span>
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
                          <button className="icon-button" type="button" aria-label={`Delete ${lesson.title}`} onClick={() => deleteLesson(lesson)}>
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted-copy">
                    {user ? "Saved lessons from your account will appear here." : "Local saved lessons stay in this browser. Sign in to keep them across devices."}
                  </p>
                )}
                {user && savedLessons.length < lessonPage.total ? (
                  <button className="ghost-button" disabled={lessonsLoading} type="button" onClick={loadMoreLessons}>
                    {lessonsLoading ? "Loading..." : `Load more (${savedLessons.length}/${lessonPage.total})`}
                  </button>
                ) : null}

                {filteredSampleLessons.length ? (
                  <div style={{ marginTop: "28px" }}>
                    <h3 style={{ fontSize: "0.78rem", color: "var(--ink-light)", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.6px", fontWeight: 800 }}>
                      Curated Sample Lessons
                    </h3>
                    <div className="lesson-list">
                      {filteredSampleLessons.map((lesson) => (
                        <article className={`lesson-card ${lesson.id === activeLessonId ? "active" : ""}`} key={lesson.id}>
                          <button className="lesson-open" type="button" onClick={() => void openLesson(lesson)}>
                            <strong>{lesson.title}</strong>
                            <span>{lessonMeta(lesson)}</span>
                          </button>
                          <div className="lesson-actions" style={{ paddingRight: "8px" }}>
                            <span className="pill" style={{ background: "rgba(0, 109, 119, 0.08)", color: "var(--accent-teal)", fontSize: "0.68rem", fontWeight: 800 }}>Demo</span>
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}

              {librarySection === "guide" ? (
                <section className="guide-panel" aria-labelledby="guide-title">
                  <div className="guide-hero">
                    <span className="pill">Turkish flow</span>
                    <h2 id="guide-title">Turkish Learning Guide</h2>
                    <p>
                      Turkish feels less scrambled when you stop chasing English word order. Start from the verb,
                      read suffixes as data, and practice speaking in ready-made Turkish chunks.
                    </p>
                  </div>
                  {currentGuideSection ? (
                    <div className="guide-section" key={currentGuideSection.title}>
                      <div className="guide-section-heading">
                        <h3>{currentGuideSection.title}</h3>
                        <p>{currentGuideSection.intro}</p>
                      </div>
                      <div className="guide-grid">
                        {currentGuideSection.strategies.map((strategy) => (
                          <article className="guide-card" key={strategy.title}>
                            <h4>{strategy.title}</h4>
                            <p>{strategy.purpose}</p>
                            <dl>
                              <div>
                                <dt>Practice</dt>
                                <dd>{strategy.drill}</dd>
                              </div>
                              <div>
                                <dt>Example</dt>
                                <dd>
                                  <code>{strategy.example}</code>
                                </dd>
                              </div>
                              <div>
                                <dt>Notice</dt>
                                <dd>{strategy.hint}</dd>
                              </div>
                            </dl>
                          </article>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {pageControls("Guide", pagedGuideSections.page, pagedGuideSections.totalPages, setGuidePage)}
                </section>
              ) : null}
            </div>
          </section>
        ) : null}

        {appScreen === "practice" ? (
          <section className="panel tab-panel">
            <div className="panel-body tab-body">
              <section className="practice-panel" aria-label="Practice like a game">
                {practiceError ? <div className="error">{practiceError}</div> : null}

                {!practiceSession ? (
                  <>
                    <div className="practice-hero">
                      <div>
                        <span className="pill">Lesson arcade</span>
                        <h2 id="practice-title">Practice like a game</h2>
                        <p>Builds interactive drills from the Turkish words, phrases, examples, and textbook sections you upload.</p>
                      </div>
                      <div className="practice-score-strip">
                        <span>
                          <Trophy size={15} />
                          {practiceProgress.xp} XP
                        </span>
                        <span>
                          <Flame size={15} />
                          {practiceStreak} streak
                        </span>
                        <span>
                          <Heart size={15} />
                          {practiceHearts}
                        </span>
                      </div>
                    </div>

                    {!result?.vocabulary_cards.length ? (
                      <div style={{ display: "grid", gap: "14px", width: "100%" }}>
                        <div className="practice-empty" style={{ minHeight: "150px", padding: "18px" }}>
                          <Gamepad2 size={32} />
                          <strong>Start Practicing Instantly</strong>
                          <p>
                            Select one of our default curated practice resources below to start playing matching, listening, and sentence building games in one click:
                          </p>
                        </div>
                        <div className="lesson-list">
                          {SAMPLE_LESSONS.map((lesson) => (
                            <article className="lesson-card" key={lesson.id} style={{ background: "rgba(255, 255, 255, 0.76)", padding: "12px", border: "1px solid rgba(0, 109, 119, 0.12)" }}>
                              <button
                                className="lesson-open"
                                type="button"
                                onClick={() => {
                                  void openLesson(lesson, "practice");
                                  window.setTimeout(() => {
                                    startPractice("mix");
                                  }, 80);
                                }}
                                style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", width: "100%", textAlign: "left", background: "none", border: 0, padding: 0 }}
                              >
                                <strong style={{ fontSize: "0.95rem", color: "var(--accent-teal)" }}>{lesson.title}</strong>
                                <span style={{ fontSize: "0.76rem", color: "var(--ink-light)", marginTop: "4px" }}>
                                  {lesson.result?.vocabulary_cards.length} cards · Tap to practice now
                                </span>
                              </button>
                            </article>
                          ))}
                        </div>
                        <div className="practice-empty" style={{ border: "1px dashed var(--panel-line)", minHeight: "110px", padding: "14px" }}>
                          <FileText size={20} />
                          <strong>Have your own vocabulary lists?</strong>
                          <p>Go to the **Learn** tab, paste a list of words or upload an image/document, and Türkçe Hoca will generate games for you.</p>
                        </div>
                      </div>
                    ) : null}

                    {result?.vocabulary_cards.length ? (
                      <div className="practice-start-card">
                        <div>
                          <span className="pill">{result.study_level}</span>
                          <h3>{result.vocabulary_cards.length} cards ready</h3>
                          <p>
                            {activeLessonId
                              ? "Progress for this saved lesson syncs to your account."
                              : user
                                ? "Save this lesson to sync practice progress across devices; draft progress stays local for now."
                                : "Draft practice progress stays in this browser until you sign in and save the lesson."}
                          </p>
                        </div>
                        <button className="primary-button" disabled={practiceLoading} type="button" onClick={() => startPractice(practiceMode)}>
                          {practiceLoading ? <Loader2 className="spin" size={18} /> : <Gamepad2 size={18} />}
                          Start Practice
                        </button>
                      </div>
                    ) : null}

                    <div className="practice-modes" role="list" aria-label="Practice activities">
                      {practiceModes.map((mode) => (
                        <button
                          className={practiceMode === mode.value ? "active" : ""}
                          key={mode.value}
                          type="button"
                          onClick={() => {
                            setPracticeMode(mode.value);
                            if (result?.vocabulary_cards.length) {
                              startPractice(mode.value);
                            }
                          }}
                        >
                          <strong>{mode.label}</strong>
                          <span>{mode.description}</span>
                        </button>
                      ))}
                    </div>

                    {result?.vocabulary_cards.length ? (
                      <div style={{ marginTop: "20px", borderTop: "1px solid var(--panel-line)", paddingTop: "16px", width: "100%" }}>
                        <span style={{ fontSize: "0.74rem", color: "var(--ink-light)", display: "block", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.5px", fontWeight: 800 }}>
                          Try Curated Sample Lessons:
                        </span>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                          {SAMPLE_LESSONS.map((lesson) => (
                            <button
                              key={lesson.id}
                              type="button"
                              className={`ghost-button ${activeLessonId === lesson.id ? "active" : ""}`}
                              onClick={() => {
                                void openLesson(lesson, "practice");
                              }}
                              style={{
                                fontSize: "0.76rem",
                                minHeight: "34px",
                                padding: "0 10px",
                                background: activeLessonId === lesson.id ? "rgba(0, 109, 119, 0.08)" : "rgba(255, 255, 255, 0.5)",
                                borderColor: activeLessonId === lesson.id ? "var(--accent-teal)" : "rgba(30, 30, 36, 0.08)"
                              }}
                              disabled={activeLessonId === lesson.id}
                            >
                              {lesson.title.replace(" (İsimler)", "").replace(" (Günlük Konuşma)", "").replace(" (Restoranda)", "")}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </>
                ) : null}

                {practiceSession ? (
                  <div className="practice-game" ref={practiceGameRef} tabIndex={-1}>
                    <div className="practice-game-top">
                      <div>
                        <span className="pill">{practiceSession.level}</span>
                        <h3>{practiceComplete ? "Practice complete" : currentPracticeQuestion?.title}</h3>
                        <p>{practiceSession.topic || practiceSession.title}</p>
                      </div>
                      <div className="practice-game-actions">
                        <button className="ghost-button" type="button" onClick={() => { setPracticeSession(null); resetPracticeInteraction(); }}>
                          <ArrowLeft size={16} />
                          Exit
                        </button>
                        <button className="ghost-button" type="button" onClick={() => startPractice(practiceMode)}>
                          <RefreshCw size={16} />
                          Restart
                        </button>
                      </div>
                    </div>

                    <div className="practice-progressbar" aria-label="Practice progress">
                      <span style={{ width: `${practiceProgressPercent}%` }} />
                    </div>

                    <div className="practice-stats">
                      <span>{Math.min(practiceIndex + 1, practiceSession.questions.length)} / {practiceSession.questions.length}</span>
                      <span>{practiceRoundCorrect} correct</span>
                      <span>{practiceMistakes} misses</span>
                      <span>{progressAccuracy(practiceProgress)}% lifetime</span>
                    </div>

                    {practiceComplete ? (
                      <div className="practice-complete">
                        <Trophy size={42} />
                        <h3>{practiceHearts <= 0 ? "Round over. Review time." : "You finished the run."}</h3>
                        <p>
                          You answered {practiceRoundCorrect} question{practiceRoundCorrect === 1 ? "" : "s"} correctly in this run.
                          {practiceProgress.missedCardIds.length ? " Missed cards are queued for Boss Review." : " No missed cards are waiting right now."}
                        </p>
                        <div className="practice-complete-actions">
                          <button className="primary-button" type="button" onClick={retryPracticeMisses}>
                            <Trophy size={18} />
                            {practiceProgress.missedCardIds.length ? "Retry missed" : "Play again"}
                          </button>
                          <button
                            className="ghost-button"
                            type="button"
                            onClick={() => {
                              setPracticeSession(null);
                              resetPracticeInteraction();
                            }}
                          >
                            Back to activities
                          </button>
                        </div>
                      </div>
                    ) : currentPracticeQuestion ? (
                      <div className={`practice-question practice-${currentPracticeQuestion.activity}`}>
                        <div className="practice-question-head">
                          <div>
                            <span>{currentPracticeQuestion.instruction}</span>
                            <h3>{currentPracticeQuestion.prompt}</h3>
                          </div>
                          <button className="icon-button" type="button" aria-label="Play question audio" onClick={() => speakPracticeQuestion(currentPracticeQuestion)}>
                            <Volume2 size={17} />
                          </button>
                        </div>

                        {currentPracticeQuestion.activity === "match" ? (
                          <div className="match-board">
                            <div>
                              <strong>Turkish</strong>
                              {currentPracticeQuestion.matchPairs.map((pair) => (
                                <button
                                  className={`${practiceMatchedIds.includes(pair.id) ? "matched" : ""} ${practiceSelectedMatch?.side === "turkish" && practiceSelectedMatch.id === pair.id ? "selected" : ""}`}
                                  disabled={practiceMatchedIds.includes(pair.id)}
                                  key={pair.id}
                                  type="button"
                                  onClick={() => choosePracticeMatch("turkish", pair.id)}
                                >
                                  {pair.turkish}
                                </button>
                              ))}
                            </div>
                            <div>
                              <strong>Meaning</strong>
                              {currentPracticeQuestion.choices.map((choice) => (
                                <button
                                  className={`${practiceMatchedIds.includes(choice.id) ? "matched" : ""} ${practiceSelectedMatch?.side === "translation" && practiceSelectedMatch.id === choice.id ? "selected" : ""}`}
                                  disabled={practiceMatchedIds.includes(choice.id)}
                                  key={choice.id}
                                  type="button"
                                  onClick={() => choosePracticeMatch("translation", choice.id)}
                                >
                                  {choice.text}
                                </button>
                              ))}
                            </div>
                          </div>
                        ) : null}

                        {["listen", "blank", "boss"].includes(currentPracticeQuestion.activity) ? (
                          <div className="choice-grid">
                            {currentPracticeQuestion.choices.map((choice, index) => (
                              <button key={choice.id} type="button" onClick={() => answerPracticeChoice(choice.text)}>
                                <span>{index + 1}</span>
                                {choice.text}
                              </button>
                            ))}
                          </div>
                        ) : null}

                        {currentPracticeQuestion.activity === "recall" ? (
                          <div className="recall-stack">
                            <input
                              aria-label="Type the Turkish answer"
                              placeholder="Type Turkish, or pick a card below"
                              type="text"
                              value={practiceTypedAnswer}
                              onChange={(event) => setPracticeTypedAnswer(event.target.value)}
                            />
                            <button className="ghost-button" disabled={!practiceTypedAnswer.trim()} type="button" onClick={submitTypedPracticeAnswer}>
                              Check typed answer
                            </button>
                            <div className="choice-grid">
                              {currentPracticeQuestion.choices.map((choice, index) => (
                                <button key={choice.id} type="button" onClick={() => answerPracticeChoice(choice.text)}>
                                  <span>{index + 1}</span>
                                  {choice.text}
                                </button>
                              ))}
                            </div>
                          </div>
                        ) : null}

                        {currentPracticeQuestion.activity === "sentence" || currentPracticeQuestion.activity === "chunk" ? (
                          <div className="builder-stack">
                            <div className="builder-answer" aria-label="Built answer">
                              {practiceBuiltParts.length ? (
                                practiceBuiltParts.map((part) => (
                                  <button key={part.id} type="button" onClick={() => removePracticePart(part)}>
                                    {part.text}
                                  </button>
                                ))
                              ) : (
                                <span>Tap pieces to build the answer.</span>
                              )}
                            </div>
                            <div className="choice-grid word-bank">
                              {currentPracticeQuestion.choices.map((choice, index) => (
                                <button
                                  disabled={practiceBuiltParts.some((part) => part.id === choice.id)}
                                  key={choice.id}
                                  type="button"
                                  onClick={() => choosePracticePart(choice)}
                                >
                                  <span>{index + 1}</span>
                                  {choice.text}
                                </button>
                              ))}
                            </div>
                            <button className="primary-button" disabled={!practiceBuiltParts.length} type="button" onClick={submitBuiltPracticeAnswer}>
                              Check order
                            </button>
                          </div>
                        ) : null}

                        {practiceFeedback ? (
                          <div className={`practice-feedback ${practiceFeedback.correct ? "correct" : "wrong"}`} role="status">
                            {practiceFeedback.correct ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
                            <div>
                              <strong>{practiceFeedback.message}</strong>
                              <p>
                                Answer: <b>{currentPracticeQuestion.answer}</b>
                                {currentPracticeQuestion.learnerNote ? ` · ${currentPracticeQuestion.learnerNote}` : ""}
                              </p>
                              {currentPracticeQuestion.exampleTr ? (
                                <span>{currentPracticeQuestion.exampleTr} — {currentPracticeQuestion.exampleTranslation}</span>
                              ) : null}
                            </div>
                          </div>
                        ) : null}

                        {practiceCanContinue ? (
                          <button className="primary-button practice-continue" type="button" onClick={nextPracticeQuestion}>
                            Continue
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>
            </div>
          </section>
        ) : null}

        {appScreen === "audio" ? (
          <section className="panel tab-panel">
            <div className="section-tabs" role="tablist" aria-label="Audio controls">
              <button className={audioSection === "playback" ? "active" : ""} type="button" onClick={() => setAudioSection("playback")}>
                <Play size={16} />
                Playback
              </button>
              <button className={audioSection === "voice" ? "active" : ""} type="button" onClick={() => setAudioSection("voice")}>
                <Volume2 size={16} />
                Voice
              </button>
              <button className={audioSection === "speed" ? "active" : ""} type="button" onClick={() => setAudioSection("speed")}>
                <RefreshCw size={16} />
                Speed
              </button>
              <button className={audioSection === "now" ? "active" : ""} type="button" onClick={() => setAudioSection("now")}>
                <Headphones size={16} />
                Now
              </button>
            </div>
            <div className="panel-body tab-body">
              <>
                {audioSection === "voice" ? (
                <div className="tts-grid">
                  <div className="field">
                    <label htmlFor="voice-select">Voice</label>
                    <select id="voice-select" value={selectedVoice} onChange={(event) => setSelectedVoice(event.target.value)}>
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
                    <select id="playback-mode" value={playbackMode} onChange={(event) => setPlaybackMode(event.target.value as PlaybackMode)}>
                      <option value="bilingual">Translation + Turkish</option>
                      <option value="turkish">Turkish only</option>
                      <option value="translation">Translation only</option>
                    </select>
                  </div>
                </div>
                ) : null}
                {audioSection === "speed" ? (
                <div className="tts-grid">
                  <div className="field">
                    <label htmlFor="playback-engine">Engine</label>
                    <select id="playback-engine" value={playbackEngine} onChange={(event) => setPlaybackEngine(event.target.value as PlaybackEngine)}>
                      <option value="browser">Browser speech</option>
                      <option value="generated">Generated audio</option>
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="speech-rate">Rate</label>
                    <div className="speed-control">
                      <div className="speed-topline">
                        <select
                          aria-label="Playback speed"
                          id="speech-rate-preset"
                          value={speechRate}
                          onChange={(event) => changeSpeechRate(Number(event.target.value))}
                        >
                          {PLAYBACK_RATE_PRESETS.map((rate) => (
                            <option key={rate} value={rate}>
                              {formatPlaybackRate(rate)}
                            </option>
                          ))}
                        </select>
                        <strong>{formatPlaybackRate(speechRate)}</strong>
                      </div>
                      <input
                        aria-label="Fine playback speed"
                        id="speech-rate"
                        max={MAX_PLAYBACK_RATE}
                        min={MIN_PLAYBACK_RATE}
                        step={PLAYBACK_RATE_STEP}
                        type="range"
                        value={speechRate}
                        onChange={(event) => changeSpeechRate(Number(event.target.value))}
                      />
                    </div>
                  </div>
                </div>
                ) : null}
                {audioSection === "now" ? (
                <label className="check-row spoken-toggle" htmlFor="show-spoken-text">
                  <input
                    checked={showSpokenText}
                    id="show-spoken-text"
                    type="checkbox"
                    onChange={(event) => setShowSpokenText(event.target.checked)}
                  />
                  <span>
                    <strong>Show spoken text</strong>
                    <small>Display the exact line being read aloud.</small>
                  </span>
                </label>
                ) : null}
                {audioSection === "playback" ? (
                <div className="player-buttons">
                  <button
                    aria-label={`Play ${readableSource.label.toLowerCase()} aloud`}
                    className="ghost-button play-button"
                    disabled={!readableSource.text}
                    title={result ? "Read the current study note or listen-practice text" : "Read the text currently in the input box"}
                    type="button"
                    onClick={speak}
                  >
                    <Play size={18} />
                    {readableSource.label}
                  </button>
                  <button
                    className="ghost-button"
                    disabled={!result?.vocabulary_cards.length}
                    type="button"
                    onClick={() =>
                      startPlaybackQueue(wordPlaybackQueue(result?.vocabulary_cards ?? [], result?.target_language ?? targetLanguage, playbackMode), "Words")
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
                      startPlaybackQueue(examplePlaybackQueue(result?.vocabulary_cards ?? [], result?.target_language ?? targetLanguage, playbackMode), "Examples")
                    }
                  >
                    <Play size={18} />
                    Examples
                  </button>
                  <button className="ghost-button" disabled={!speaking} type="button" onClick={pauseOrResume}>
                    {paused ? <RefreshCw size={18} /> : <Pause size={18} />}
                    {paused ? "Resume" : "Pause"}
                  </button>
                  <button aria-label="Previous item" className="icon-button" disabled={!speaking || playbackTotal < 2} type="button" onClick={() => skipPlayback(-1)}>
                    <SkipBack size={18} />
                  </button>
                  <button aria-label="Next item" className="icon-button" disabled={!speaking || playbackTotal < 2} type="button" onClick={() => skipPlayback(1)}>
                    <SkipForward size={18} />
                  </button>
                  <button aria-label="Stop playback" className="icon-button" disabled={!speaking} type="button" onClick={stopSpeech}>
                    <Square size={18} />
                  </button>
                </div>
                ) : null}
                <div className="playback-status" aria-live="polite">
                  <span>{speaking ? playbackLabel || "Read Aloud" : "Background playback"}</span>
                  <strong>
                    {audioLoading
                      ? "Generating..."
                      : speaking
                        ? `${playbackProgress(playbackCurrent, playbackTotal)} · ${activeEngine === "generated" ? "audio" : "speech"}`
                        : ttsConfig?.configured
                          ? `${ttsConfig.provider} ready`
                          : pwaReady
                            ? "PWA ready"
                            : "Browser speech"}
                  </strong>
                </div>
                {showSpokenText ? (
                  <div className={`spoken-text-panel ${currentSpokenText ? "" : "is-idle"}`} aria-live="polite">
                    {currentSpokenText ? (
                      <>
                        <div className="spoken-text-meta">
                          <span>{currentSpokenText.title}</span>
                          <strong>{currentSpokenText.progress}</strong>
                        </div>
                        {currentSpokenText.subtitle ? <p className="spoken-text-subtitle">{currentSpokenText.subtitle}</p> : null}
                        <p className="spoken-text-current">{currentSpokenText.text}</p>
                        <span className="spoken-text-lang">{currentSpokenText.lang}</span>
                      </>
                    ) : (
                      <p className="spoken-text-empty">Start read-aloud playback to show the current spoken text here.</p>
                    )}
                  </div>
                ) : null}
                <p className="muted-copy">
                  Browser speech is the default and does not call the generated-audio API. Select Generated audio only when you want provider audio for stronger mobile background playback; it may use paid API credits.
                  Speed uses YouTube-style presets from 0.25x to 2.0x; some browser voices may clamp extreme speech rates.
                </p>
                {!ttsConfig?.configured && playbackEngine !== "browser" ? (
                  <p className="voice-warning">Generated audio is not configured. Add OpenAI TTS settings on the API, or choose Browser speech.</p>
                ) : null}
                {ttsConfig?.configured && !user ? (
                  <p className="voice-warning">Sign in to use generated audio. Browser speech still works without an account.</p>
                ) : null}
                {playbackNotice ? <p className="voice-warning">{playbackNotice}</p> : null}
                {!turkishVoices.length ? (
                  <p className="voice-warning">No Turkish browser voice is currently available. Playback will use the closest installed voice.</p>
                ) : null}
              </>
            </div>
          </section>
        ) : null}

        {appScreen === "results" ? (
          <section className="study-output-shell">
            {!result ? (
              <section className="panel empty-state">
            <div>
              <Upload size={40} />
                  <p>No study result yet.</p>
                  <button className="primary-button" type="button" onClick={() => openAppScreen("learn")}>
                    Go to Learn
                  </button>
            </div>
          </section>
        ) : (
          <>
                <section className="panel result-shell">
              <div className="panel-header">
                <div className="panel-title">
                  <FileText size={18} />
                  <h2>{activeLesson ? "Saved Lesson" : "Results"}</h2>
                </div>
                {activeLesson ? (
                  <button className="ghost-button" type="button" style={{ minHeight: "30px", padding: "0 8px", fontSize: "12px" }} onClick={() => { setLibrarySection("lessons"); openAppScreen("library"); }}>
                    <ArrowLeft size={14} />
                    <span>Back to Library</span>
                  </button>
                ) : null}
              </div>
                  <div className="section-tabs result-tabs" role="tablist" aria-label="Result sections">
                    {resultTabs.map((tab) => (
                      <button
                        className={resultSection === tab.value ? "active" : ""}
                        disabled={tab.count === 0}
                        key={tab.value}
                        type="button"
                        onClick={() => setResultSection(tab.value)}
                      >
                        {tab.label}
                        <span>{tab.count}</span>
                      </button>
                    ))}
                  </div>
              <div className="panel-body">
                    {resultSection === "overview" ? (
                      <>
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
                {result.extraction_warning ? <div className="warning">{result.extraction_warning}</div> : null}
                        <div className="flow-actions">
                          <button className="ghost-button" type="button" onClick={() => openAppScreen("learn")}>
                            <ArrowLeft size={18} />
                            Back to Learn
                          </button>
                          <button className="ghost-button" type="button" onClick={() => setResultSection("vocabulary")}>
                            Vocabulary
                          </button>
                          <button className="ghost-button" disabled={!result.vocabulary_cards.length} type="button" onClick={() => startPractice(practiceMode)}>
                            Practice these
                          </button>
                          <button className="ghost-button" type="button" onClick={() => openAppScreen("audio")}>
                            Read aloud
                          </button>
                        </div>
                      </>
                    ) : null}

                    {resultSection === "textbook" ? (
                      <>
                        {result.textbook_warning ? <div className="warning">{result.textbook_warning}</div> : null}
                        {pagedTextbookSections.items.length ? (
                          <div className="textbook-section-list paged-list">
                            {pagedTextbookSections.items.map((section, index) => (
                              <article className="textbook-section" key={`${section.title}-${pagedTextbookSections.start + index}`}>
                                <div className="textbook-section-head">
                                  <div>
                                    <span className="pill">{section.section_type}</span>
                                    <h3>{section.title}</h3>
                                    <p>{section.topic}</p>
                                  </div>
                                  <span>{section.source_pages || section.level}</span>
                                </div>
                                <p>{section.summary}</p>
                                <div className="textbook-grid">
                                  <div>
                                    <strong>Key vocabulary</strong>
                                    <ul>
                                      {section.key_vocabulary.slice(0, 8).map((item) => (
                                        <li key={item}>{item}</li>
                                      ))}
                                    </ul>
                                  </div>
                                  <div>
                                    <strong>Grammar focus</strong>
                                    <ul>
                                      {section.grammar_focus.slice(0, 6).map((item) => (
                                        <li key={item}>{item}</li>
                                      ))}
                                    </ul>
                                  </div>
                                </div>
                                <div className="example-block">
                                  <strong>Translation / meaning</strong>
                                  <span>{section.translation}</span>
                                </div>
                                <div className="textbook-practice">
                                  <strong>Practice aligned to this section</strong>
                                  <ul>
                                    {section.practice.slice(0, 4).map((item) => (
                                      <li key={item}>{item}</li>
                                    ))}
                                  </ul>
                                </div>
                              </article>
                            ))}
                          </div>
                        ) : (
                          <p className="muted-copy">No textbook sections were detected for this input.</p>
                        )}
                        {pageControls("Textbook", pagedTextbookSections.page, pagedTextbookSections.totalPages, setTextbookPage)}
                      </>
                    ) : null}

                    {resultSection === "vocabulary" ? (
                      <>
                        {result.vocabulary_warning ? <div className="warning">{result.vocabulary_warning}</div> : null}
                        <div className="filters">
                          <div className="search-box">
                            <Search size={16} />
                            <input
                              aria-label="Search vocabulary"
                              placeholder="Search words, translations, examples"
                              value={search}
                              onChange={(event) => {
                                setSearch(event.target.value);
                                setVocabPage(1);
                              }}
                            />
                            {search ? (
                              <button className="clear-search-button" type="button" aria-label="Clear search" onClick={() => { setSearch(""); setVocabPage(1); }}>
                                <XCircle size={16} />
                              </button>
                            ) : null}
                          </div>
                          <select
                            aria-label="Filter by type"
                            value={typeFilter}
                            onChange={(event) => {
                              setTypeFilter(event.target.value);
                              setVocabPage(1);
                            }}
                          >
                            {cardTypes.map((type) => (
                              <option key={type} value={type}>
                                {type === "all" ? "All types" : type}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="list-count">
                          Showing {pagedCards.end ? pagedCards.start + 1 : 0}-{pagedCards.end} of {filteredCards.length}
                        </div>
                        <div className="vocab-list paged-list">
                          {pagedCards.items.map((card, index) => (
                            <article className="vocab-card" key={`${card.turkish}-${pagedCards.start + index}`}>
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
                                    onClick={() => speakSegments(wordSegments(card, result.target_language, playbackMode))}
                                  >
                                    <Play size={16} />
                                  </button>
                                  <button
                                    aria-label={`Play example for ${card.turkish}`}
                                    className="icon-button"
                                    type="button"
                                    title={formatPair(card.example_tr, card.example_translation)}
                                    onClick={() => speakSegments(exampleSegments(card, result.target_language, playbackMode))}
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
                        {pageControls("Vocabulary", pagedCards.page, pagedCards.totalPages, setVocabPage)}
                      </>
                    ) : null}

                    {resultSection === "units" ? (
                      <>
                        <div className="unit-list paged-list">
                          {pagedUnits.items.map((unit, index) => (
                            <div className="unit" key={`${unit.kind}-${pagedUnits.start + index}`}>
                              <span>{unit.kind}</span>
                              <p>{unit.text}</p>
                            </div>
                          ))}
                        </div>
                        {pageControls("Units", pagedUnits.page, pagedUnits.totalPages, setUnitPage)}
                      </>
                    ) : null}

                    {resultSection === "note" ? <pre className="note">{result.note}</pre> : null}
              </div>
            </section>
          </>
        )}
      </section>
        ) : null}
      </section>
    </main>
  );
}
