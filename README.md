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
- 🔊 **Language-aware text-to-speech** — `/read` reads Turkish or other languages aloud with voice/rate controls
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

The web app supports text input, file uploads, CEFR level selection, target-language selection, extracted-text preview, generated study notes, detected study units, and browser text-to-speech controls.

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

---

## 🗂️ Project Structure

```
turkish-tutor/
├── tutor.py        — Main interactive CLI tutor (Gemini-powered)
├── config.py       — Teaching strategies, CEFR levels, system prompt
├── content_intelligence.py — Text/PDF/DOCX/image extraction and CEFR study prompt helpers
├── speech.py       — Language-aware TTS voice discovery and playback
├── dataset.py      — Knowledge base pipeline (vocabulary, grammar, Q&A)
├── evaluate.py     — 4-metric evaluator (used by autoresearch swarm)
├── swarm.py        — Multi-agent strategy optimizer
├── tests/          — Unit tests for extraction and speech helpers
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
