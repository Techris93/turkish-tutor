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
- 🤖 **Autoresearch loop** — AI agents experiment with different teaching strategies to improve scores
- 📊 **Evaluation pipeline** — 4-metric scoring: accuracy, pedagogy, Turkish correctness, composite

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
# Get a free key at: https://aistudio.google.com

# 3. Build the knowledge base
python dataset.py

# 4. Start the tutor!
python tutor.py
```

---

## 💬 Tutor Commands

| Command | Description |
|---|---|
| `/level` | Change CEFR level (A1/A2/B1/B2/C1/C2) |
| `/exercise` | Get a practice exercise |
| `/vocab` | Vocabulary flashcard drill |
| `/progress` | Show session stats |
| `/help` | Show all commands |
| `/quit` | End session |

---

## 🗂️ Project Structure

```
turkish-tutor/
├── tutor.py        — Main interactive CLI tutor (Gemini-powered)
├── config.py       — Teaching strategies, CEFR levels, system prompt
├── dataset.py      — Knowledge base pipeline (vocabulary, grammar, Q&A)
├── evaluate.py     — 4-metric evaluator (used by autoresearch swarm)
├── swarm.py        — Multi-agent strategy optimizer
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
