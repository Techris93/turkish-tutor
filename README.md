# 🇹🇷 Türkçe Hoca — AI Turkish Language Tutor

An adaptive AI language tutor for Turkish, powered by Google Gemini. Uses the **autoresearch swarm loop** to continuously improve teaching strategies.

[![CEFR Levels](https://img.shields.io/badge/CEFR-A1%20→%20C2-blue)](https://coe.int/en/web/common-european-framework-reference-languages)
[![Powered by Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange)](https://aistudio.google.com)

---

## ✨ Features

- 🎓 **CEFR-adaptive teaching** — A1 through C2, adjusts explanation depth and language
- 📚 **Comprehensive knowledge base** — Grammar rules, vocabulary lists, cultural notes
- 🔤 **Interactive exercises** — Fill-in-the-blank, translation, error-spotting
- 🃏 **Vocabulary flashcard drill** — `/vocab` command
- 🖼️ **Image/PDF/text study intake** — `/study` extracts Turkish from typed text, images, PDFs, DOCX, and text files
- 🌍 **Translation + CEFR examples** — translates extracted words/phrases/sentences and generates A1-C2 practice lines
- 🧾 **Vocabulary cards from photos** — splits OCR tables into individual words/phrases, preserves compounds, and creates one translated example card per item
- 🔊 **Bilingual text-to-speech** — reads Turkish words/examples alone, translations alone, or Turkish followed by the translation
- 💾 **Account saved lessons** — saves generated study sessions to a user account so learners can revisit and revise them later
- 🔐 **Production auth basics** — email/password auth, SMTP password reset, Google/GitHub OAuth, and rate limiting
- 🤖 **Autoresearch loop** — AI agents experiment with different teaching strategies to improve scores
- 📊 **Evaluation pipeline** — 4-metric scoring: accuracy, pedagogy, Turkish correctness, composite

---

## 🚀 Quick Start

```bash
# 1. Create a local environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Optional but recommended for image OCR on macOS
brew install tesseract tesseract-lang

# 3. Set your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
# Get a free key at: https://aistudio.google.com

# 4. Build the knowledge base
python dataset.py

# 5. Start the tutor!
python tutor.py
```

## Web App

Run the FastAPI backend and Next.js frontend in two terminals:

```bash
# Terminal 1: API
source .venv/bin/activate
uvicorn api:app --reload --port 8000

# Terminal 2: Next.js
cd web
npm install
npm run dev
```

Then open http://localhost:3000.

The frontend calls `http://127.0.0.1:8000` by default. To use a different API URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The web app supports text input, file uploads, CEFR level selection, target-language selection, extracted-text preview, generated study notes, detected study units, structured vocabulary cards, browser text-to-speech controls, authentication, and account-backed saved lessons.

### Local Auth And Database Setup

Saved lessons are stored through the FastAPI backend. Local development uses SQLite by default:

```bash
DATABASE_URL=sqlite:///data/turkish_tutor.sqlite3
AUTH_COOKIE_SECURE=false
AUTH_COOKIE_SAMESITE=lax
```

The API creates the required tables at startup. User passwords are hashed with Argon2. Browser sessions use HTTP-only cookies, so the frontend must call the API with credentials enabled. For browsers that do not reliably keep the API cookie after a cross-subdomain OAuth redirect, the app also keeps the current session token in `sessionStorage` for that tab and sends it as `X-Session-Token`; logout invalidates the same server-side session. If the learner checks `Remember me`, the same fallback token is also stored in `localStorage` on that device so the app can restore the account after the browser is closed.

For a local password-reset test without sending email, set:

```bash
PASSWORD_RESET_RETURN_TOKEN=true
```

This returns the reset token in the API response for development only.

To send real reset emails locally or in production, configure SMTP:

```bash
PASSWORD_RESET_BASE_URL=http://localhost:3000
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_smtp_user
SMTP_PASSWORD=your_smtp_password
SMTP_FROM_EMAIL=noreply@example.com
SMTP_USE_TLS=true
```

Reset links open the web app with a `reset_token` query parameter. The app shows the new-password form automatically when that token is present.

## Deploy On Render

This repo includes a Render Blueprint in `render.yaml` with two services and one Postgres database:

- `turkish-tutor-api`: FastAPI backend built from `Dockerfile.api`. Docker is used so image OCR has the required `tesseract-ocr`, English, and Turkish language packages.
- `turkish-tutor-web`: Next.js frontend exported as a static site from `web/out`.
- `turkish-tutor-db`: Render Postgres database for users, sessions, password reset tokens, and saved lessons.

Deploy steps:

1. Commit and push this repo branch to GitHub.
2. Open the Blueprint deeplink:
   `https://dashboard.render.com/blueprint/new?repo=https://github.com/Techris93/turkish-tutor`
3. Select the pushed branch, usually `main`.
4. Fill the required secret env var for `turkish-tutor-api`:
   - `GEMINI_API_KEY`
5. Apply the Blueprint and wait for the API, web service, and database to deploy.
6. Open `https://turkish-tutor-web.onrender.com`.

The frontend is configured with `NEXT_PUBLIC_API_URL=https://turkish-tutor-api.onrender.com`, and the API allows CORS from `https://turkish-tutor-web.onrender.com`. The API receives `DATABASE_URL` from the Blueprint-managed Postgres database, and production cookies use `AUTH_COOKIE_SECURE=true` plus `AUTH_COOKIE_SAMESITE=none`. If you rename services in Render, update those values in `render.yaml`.

Render syncs `sync: false` secrets only during initial Blueprint creation. Set or update these secrets manually in the Render Dashboard as needed:

- `GEMINI_API_KEY`
- `SMTP_HOST`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GITHUB_OAUTH_CLIENT_ID`
- `GITHUB_OAUTH_CLIENT_SECRET`

Render CLI validation, if installed and authenticated:

```bash
render blueprints validate
```

### Image Vocabulary Workflow

When an uploaded photo contains a Turkish word list or table, the backend now:

- Cleans OCR text and removes labels such as `İSİMLER`, `FİİLLER`, and table headings.
- Splits row-style OCR like `arkadaş çarşı inek mavi salon açmak` into individual vocabulary entries.
- Preserves common multi-word items such as `anneler günü`, `çocuk odası`, `doğum günü`, `cevap vermek`, and `tekrar etmek`.
- Infers a light category for each item: noun/unknown, verb, color/adjective, nationality, place/country, or phrase.
- Asks Gemini for strict JSON vocabulary cards, then validates and repairs the response so every detected item is still displayed.

Each vocabulary card includes the Turkish item, English translation, category, CEFR level, a level-appropriate Turkish example, the translated example, a short note, and text-to-speech text for both the word and the example.

In the frontend, use:

- Search to find a word, translation, or example.
- Type filter to focus on verbs, colors, nationalities, places, or phrases.
- The playback mode menu to choose `Turkish + translation`, `Turkish only`, or `Translation only`.
- Per-card play buttons to hear one word or one example. In bilingual mode, word playback is spoken as pairs such as `gel, come`; example playback is spoken as pairs such as `buraya gel, come here`.
- `Words` and `Examples` playback buttons to queue all detected vocabulary with the selected playback mode.

Read-aloud can use generated audio or browser text-to-speech:

- `Browser speech`: the default. It uses the browser `SpeechSynthesis` engine only and never calls the generated-audio API.
- `Generated audio`: an explicit opt-in. It uses backend-generated audio through an `HTMLAudioElement` and may use paid provider credits.

Browser text-to-speech tries to use a Turkish voice for Turkish segments and a target-language voice for translations. Available voices vary by browser and operating system, so install system voices if pronunciation quality is limited.

### Background Playback And PWA

The web app is installable as a lightweight PWA. On supported browsers, install it from the browser menu or address-bar install button, then open it from the home screen/app launcher for the best background playback behavior.

Read-aloud playback uses a queue controller with progress, previous/next controls, pause/resume/stop, and Media Session API handlers. Where the browser supports Media Session, system media controls can show the current Turkish word/example and can control play, pause, stop, previous, and next.

Important limits:

- Generated audio is the recommended mode for mobile background and locked-screen listening because it uses normal browser audio playback.
- Browser speech is still available as the free/no-key fallback.
- Desktop browsers usually allow speech to continue while the tab is hidden or the window is minimized.
- Mobile locked-screen playback is still browser and OS dependent. iOS Safari and some mobile browsers may pause or stop audio depending on install state, power settings, and autoplay rules.
- Installing the PWA can improve the chance of background controls, but it cannot override mobile OS restrictions.

### Generated Audio TTS

The API includes a provider abstraction and currently supports OpenAI TTS for production generated audio. It uses OpenAI's `audio/speech` endpoint, which supports MP3 output, built-in voices, and Turkish input text. The app discloses generated audio in the UI because OpenAI requires users to know when a voice is AI-generated.

Configure locally:

```bash
TTS_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE_TR=nova
OPENAI_TTS_VOICE_DEFAULT=alloy
OPENAI_TTS_TIMEOUT_SECONDS=45
RATE_LIMIT_TTS=120/1h
```

For tests or local smoke checks without a paid provider, use `TTS_PROVIDER=mock`. The mock provider returns a tiny generated WAV tone and must not be used as production speech.

Generated audio requires a signed-in user and is rate-limited to control provider cost. If `TTS_PROVIDER` or `OPENAI_API_KEY` is missing, the frontend shows a clear message. Normal `Play`, `Words`, `Examples`, and per-card playback use browser speech unless the learner explicitly selects `Generated audio`.

On Render, set `OPENAI_API_KEY` as a secret environment variable. The Blueprint includes non-secret defaults for `TTS_PROVIDER=openai`, `OPENAI_TTS_MODEL`, voices, timeout, and `RATE_LIMIT_TTS`.

### Saved Lessons And Accounts

The web app can save the current study result as a lesson:

- Enter or edit a lesson title in the `Saved Lessons` panel.
- Sign up or log in to save lessons to your account.
- Click `Save` to store the current extracted text, study note, CEFR level, target language, source details, and vocabulary cards in the backend database.
- Use the saved lesson list to reopen a lesson without uploading the image again or calling Gemini again.
- Rename or delete saved lessons from the lesson list. Delete asks for confirmation first.
- If older browser-only lessons exist in `localStorage`, log in and use `Import to account` to copy them into persistent storage.

When logged in, saved lessons are stored in the backend database and survive refreshes, browser restarts, and device changes for the same account. When logged out, the app can still keep temporary local lessons in the browser's `localStorage` under `turkce-hoca.saved-lessons.v1`, but those local drafts are not synced and can be removed if site data is cleared.

### Authentication

The web app includes:

- Email/password sign-up and login.
- Logout with server-side session invalidation.
- Current-user check on app load.
- HTTP-only cookie sessions stored in the database, with a tab-scoped `sessionStorage` fallback for OAuth/cookie-constrained browsers.
- Optional `Remember me` storage for personal devices; this stores the fallback session token in `localStorage` until logout.
- SMTP-backed password reset request/confirm endpoints and UI.
- Google and GitHub OAuth start/callback routes with database-backed state validation.
- In-process rate limiting for high-risk endpoints.

OAuth buttons remain disabled until the matching credentials and redirect URIs are configured:

```bash
GOOGLE_OAUTH_CLIENT_ID=
GOOGLE_OAUTH_CLIENT_SECRET=
GOOGLE_OAUTH_REDIRECT_URI=http://127.0.0.1:8000/api/auth/oauth/google/callback
GITHUB_OAUTH_CLIENT_ID=
GITHUB_OAUTH_CLIENT_SECRET=
GITHUB_OAUTH_REDIRECT_URI=http://127.0.0.1:8000/api/auth/oauth/github/callback
OAUTH_SUCCESS_REDIRECT_URL=http://localhost:3000/?oauth=success
OAUTH_ERROR_REDIRECT_URL=http://localhost:3000/?oauth=error
OAUTH_HANDOFF_MINUTES=5
```

For Render, use:

```bash
GOOGLE_OAUTH_REDIRECT_URI=https://turkish-tutor-api.onrender.com/api/auth/oauth/google/callback
GITHUB_OAUTH_REDIRECT_URI=https://turkish-tutor-api.onrender.com/api/auth/oauth/github/callback
OAUTH_SUCCESS_REDIRECT_URL=https://turkish-tutor-web.onrender.com/?oauth=success
OAUTH_ERROR_REDIRECT_URL=https://turkish-tutor-web.onrender.com/?oauth=error
```

OAuth links accounts by verified/provider email. If the email already exists, the OAuth login uses the existing account; otherwise it creates a new account.

After the provider callback, the API redirects the web app with a short-lived one-time handoff code. The frontend redeems that code, stores the returned session token in `sessionStorage` for the current browser tab, confirms `/api/auth/me`, loads saved lessons, and only then shows OAuth success. If `Remember me` was checked before starting OAuth, the redeemed session token is also stored in `localStorage`. This avoids a misleading “OAuth login completed” state when a browser does not accept the API session cookie across Render subdomains. Handoff codes are stored hashed, expire quickly, and can be used only once.

Use `Remember me` only on your own device. Logout clears both the session-only token and the remembered token. Server sessions default to `AUTH_SESSION_DAYS=30`; for a personal deployment you can raise that value, for example to `90` or `180`, in Render environment variables.

### Rate Limiting

The API includes a lightweight in-process limiter. Defaults can be overridden with:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_SIGNUP=5/1h
RATE_LIMIT_LOGIN=10/5m
RATE_LIMIT_PASSWORD_RESET=5/1h
RATE_LIMIT_PASSWORD_RESET_CONFIRM=10/1h
RATE_LIMIT_STUDY=30/1h
RATE_LIMIT_LESSON_WRITE=120/1h
RATE_LIMIT_TTS=120/1h
```

The limiter is intentionally simple for this small app and protects sign-up, login, password reset, OAuth, lesson writes, Gemini-backed study requests, and generated TTS requests. If you run multiple API instances or open this app to broader public traffic, replace it with Redis or another shared rate limiter.

Current production setup still required:

- Add real SMTP credentials before password reset emails can be delivered.
- Add Google/GitHub OAuth app credentials and exact redirect URLs before OAuth buttons become active.
- Add `OPENAI_API_KEY` before generated audio TTS becomes available.
- Keep `PASSWORD_RESET_RETURN_TOKEN=false` outside local development.

After pushing changes to the branch connected to Render, Render should redeploy the API/static web services automatically from the Blueprint.

---

## 💬 Tutor Commands

| Command | Description |
|---|---|
| `/level` | Change CEFR level (A1/A2/B1/B2/C1/C2) |
| `/exercise` | Get a practice exercise |
| `/vocab` | Vocabulary flashcard drill |
| `/study <text-or-path>` | Extract, translate, explain, and generate level-matched examples |
| `/read <text>` | Read text aloud with local text-to-speech |
| `/voices [language]` | List available local voices |
| `/progress` | Show session stats |
| `/help` | Show all commands |
| `/quit` | End session |

### Study Intake Examples

```bash
# In the interactive tutor:
/study Merhaba, nasılsın?
/study --level B1 --target English /Users/me/Desktop/turkish-textbook.pdf
/study --target Spanish /Users/me/Desktop/menu-photo.jpg
/read last
/read --lang tr --rate 160 Merhaba, bugün Türkçe çalışıyoruz.
/voices tr

# One-shot analysis from the shell:
python tutor.py --level A2 --study /Users/me/Desktop/worksheet.pdf
```

Supported intake:

- Typed or pasted Turkish words, phrases, and longer text
- Plain text/Markdown/CSV/JSON/SRT files
- PDFs via `pypdf`
- DOCX files via `python-docx`
- Images/photos/screenshots via `pytesseract` + the Tesseract OCR app

Text-to-speech uses macOS `say` automatically on macOS, including installed Turkish voices. On other platforms, install `pyttsx3` for a basic local fallback. Use `/voices tr` to see whether a Turkish voice is installed.

### Verification

```bash
source .venv/bin/activate
python -m py_compile api.py auth_storage.py email_delivery.py oauth_flow.py rate_limit.py tts_provider.py tutor.py config.py dataset.py evaluate.py swarm.py content_intelligence.py speech.py vocabulary_cards.py
python -m unittest discover -s tests

cd web
npm run test
npm run lint
npm run build
```

---

## 🗂️ Project Structure

```
turkish-tutor/
├── tutor.py        — Main interactive CLI tutor (Gemini-powered)
├── api.py          — FastAPI backend for the web app
├── auth_storage.py — SQLAlchemy auth, sessions, password reset tokens, and saved lessons
├── email_delivery.py — SMTP password reset email delivery
├── oauth_flow.py   — Google/GitHub OAuth provider exchange helpers
├── rate_limit.py   — Small in-process API rate limiter
├── tts_provider.py  — Generated-audio TTS provider abstraction and OpenAI TTS implementation
├── config.py       — Teaching strategies, CEFR levels, system prompt
├── content_intelligence.py — Text/PDF/DOCX/image extraction and CEFR study prompt helpers
├── vocabulary_cards.py — Structured vocabulary-card JSON parsing and fallbacks
├── speech.py       — Language-aware TTS voice discovery and playback
├── dataset.py      — Knowledge base pipeline (vocabulary, grammar, Q&A)
├── evaluate.py     — 4-metric evaluator (used by autoresearch swarm)
├── swarm.py        — Multi-agent strategy optimizer
├── tests/          — Unit tests for extraction and speech helpers
├── web/            — Next.js frontend
├── data/
│   ├── knowledge.json    — Turkish language knowledge base
│   ├── test_qa.json      — Test Q&A dataset (20 CEFR-leveled questions)
│   └── eval_results/     — Evaluation scores per experiment branch
└── .env.example    — Environment variable template
```

---

## 🔬 Autoresearch Loop

The swarm experiments with different teaching strategies to maximize pedagogical quality:

```bash
# Spawn agent branches (each tries a different teaching strategy)
python swarm.py --spawn 3

# On each branch, run evaluation to get a score
python evaluate.py --verbose

# See which strategy performs best
python swarm.py --leaderboard

# Promote the winning config to main
python swarm.py --adopt experiment/agent-1 --confirm
```

**Experiment directions** include: Socratic method, mnemonic-heavy teaching,
strict CEFR calibration, cultural immersion, spaced repetition, error-first teaching,
comprehensible i+1 input, and morpheme-first grammar.

---

## 📊 Evaluation Metrics

| Metric | Weight | Description |
|---|---|---|
| **Accuracy** | 50% | Did the tutor answer correctly vs gold standard? |
| **Pedagogy** | 30% | Is the explanation clear, encouraging, level-appropriate? |
| **Turkish Correctness** | 20% | Is the Turkish in the response grammatically correct? |
| **Composite** | — | Weighted average, used for branch comparison |

---

## 📦 Dataset Sources

The built-in knowledge base includes:

- **10 grammar rules** with full examples (vowel harmony, suffixes, tenses, word order)
- **~100 vocabulary items** across A1–B2 CEFR levels (greetings, numbers, family, work, etc.)
- **20 test Q&A pairs** with CEFR-level labels and difficulty ratings
- **Optional**: TQuAD (Turkish Q&A from HuggingFace) with `python dataset.py --fetch-hf`

---

## 🇹🇷 Why Turkish is Unique

Turkish is an **agglutinative SOV language** — this makes it a rewarding challenge:
- Suffixes chain together: `evlerimizden` = from our houses (ev + ler + imiz + den)
- **Vowel harmony**: suffixes change to match the vowels in the root
- Verb always comes **last**: Ben Türkçe öğreniyorum (I Turkish am-learning)

---

*Part of the [Antigravity](https://github.com/Techris93) project suite.*
