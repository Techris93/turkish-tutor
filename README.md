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
- 💾 **Saved web lessons** — saves generated study sessions locally so learners can revisit and revise them later
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

The web app supports text input, file uploads, CEFR level selection, target-language selection, extracted-text preview, generated study notes, detected study units, structured vocabulary cards, browser text-to-speech controls, and saved lessons.

## Deploy On Render

This repo includes a Render Blueprint in `render.yaml` with two services:

- `turkish-tutor-api`: FastAPI backend built from `Dockerfile.api`. Docker is used so image OCR has the required `tesseract-ocr`, English, and Turkish language packages.
- `turkish-tutor-web`: Next.js frontend exported as a static site from `web/out`.

Deploy steps:

1. Commit and push this repo branch to GitHub.
2. Open the Blueprint deeplink:
   `https://dashboard.render.com/blueprint/new?repo=https://github.com/Techris93/turkish-tutor`
3. Select the pushed branch, usually `main`.
4. Fill the required secret env var for `turkish-tutor-api`:
   - `GEMINI_API_KEY`
5. Apply the Blueprint and wait for both services to deploy.
6. Open `https://turkish-tutor-web.onrender.com`.

The frontend is configured with `NEXT_PUBLIC_API_URL=https://turkish-tutor-api.onrender.com`, and the API allows CORS from `https://turkish-tutor-web.onrender.com`. If you rename either service in Render, update those two values in `render.yaml`.

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

Browser text-to-speech uses `SpeechSynthesis`. The app tries to use a Turkish voice for Turkish segments and a target-language voice for translations. Available voices vary by browser and operating system, so install system voices if pronunciation quality is limited.

### Saved Lessons

The web app can save the current study result as a lesson:

- Enter or edit a lesson title in the `Saved Lessons` panel.
- Click `Save` to store the current extracted text, study note, CEFR level, target language, source details, and vocabulary cards.
- Use the saved lesson list to reopen a lesson without uploading the image again or calling Gemini again.
- Rename or delete saved lessons from the lesson list. Delete asks for confirmation first.

Saved lessons are stored in the browser's `localStorage` under `turkce-hoca.saved-lessons.v1`. They survive refreshes and browser restarts on the same device/browser, but they are not synced across devices and will be removed if site data is cleared.

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
python -m py_compile api.py tutor.py config.py dataset.py evaluate.py swarm.py content_intelligence.py speech.py vocabulary_cards.py
python -m unittest discover -s tests

cd web
npm run lint
npm run build
```

---

## 🗂️ Project Structure

```
turkish-tutor/
├── tutor.py        — Main interactive CLI tutor (Gemini-powered)
├── api.py          — FastAPI backend for the web app
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
