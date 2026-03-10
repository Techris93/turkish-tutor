"""
Turkish Agent Tutor — Main Tutor Interface
Gemini-powered interactive Turkish language tutor with CEFR-level adaptive teaching.

Usage:
    python tutor.py                    # Start interactive session
    python tutor.py --level A2         # Start at specific CEFR level
    python tutor.py --topic grammar    # Start with a specific topic
    python tutor.py --exercise         # Jump straight to exercises
"""

import os
import sys
import json
import argparse
import asyncio
from datetime import datetime
from typing import Optional, List, Dict

from dotenv import load_dotenv
load_dotenv()

# ─── Terminal Colors ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
BG_BLUE   = "\033[44m"
BG_GREEN  = "\033[42m"

DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.json")
SESSIONS_FILE  = os.path.join(DATA_DIR, "sessions.json")


# ─── Gemini Client ────────────────────────────────────────────────────────────

GEMINI_AVAILABLE = False
_client = None

def _init_gemini():
    global GEMINI_AVAILABLE, _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return False
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
        return True
    except ImportError:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            _client = genai_legacy.GenerativeModel("gemini-2.5-flash")
            GEMINI_AVAILABLE = True
            return True
        except ImportError:
            return False


async def ask_gemini(prompt: str, temperature: float = 0.4) -> str:
    """Send prompt to Gemini, return response text."""
    if not GEMINI_AVAILABLE:
        return "[Gemini not available — check GEMINI_API_KEY in .env]"
    try:
        loop = asyncio.get_event_loop()
        try:
            from google import genai as _genai
            from google.genai import types
            response = await loop.run_in_executor(
                None,
                lambda: _client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=600,
                    )
                )
            )
        except (ImportError, AttributeError):
            response = await loop.run_in_executor(
                None,
                lambda: _client.generate_content(prompt)
            )
        return response.text.strip()
    except Exception:
        return "[Error: Could not generate response. Please try again.]"


# ─── Knowledge Base ───────────────────────────────────────────────────────────

def load_knowledge() -> List[Dict]:
    """Load the Turkish knowledge base."""
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"\n{YELLOW}⚠️  No knowledge base found. Building it now...{RESET}")
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'dataset.py')],
            check=False
        )
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("knowledge_base", [])
    return []


# ─── Session Management ────────────────────────────────────────────────────────

class TutorSession:
    """Manages a user's tutoring session: CEFR level, progress, history."""

    def __init__(self, username: str = "learner", cefr_level: str = "A1"):
        self.username = username
        self.cefr_level = cefr_level
        self.history: List[Dict] = []
        self.exchange_count = 0
        self.correct_answers = 0
        self.topics_covered: List[str] = []
        self.started_at = datetime.now().isoformat()

    def add_exchange(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if role == "user":
            self.exchange_count += 1

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        sessions = []
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, encoding="utf-8") as f:
                sessions = json.load(f)
        sessions.append({
            "username": self.username,
            "cefr_level": self.cefr_level,
            "exchanges": self.exchange_count,
            "correct": self.correct_answers,
            "topics": self.topics_covered,
            "started_at": self.started_at,
            "ended_at": datetime.now().isoformat(),
        })
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions[-100:], f, indent=2, ensure_ascii=False)


# ─── Display Helpers ──────────────────────────────────────────────────────────

def clear_line():
    print("\r" + " " * 60 + "\r", end="", flush=True)


def print_banner():
    print(f"\n{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(f"{BOLD}{BG_BLUE}{WHITE}   🇹🇷  Türkçe Hoca — AI Turkish Language Tutor              {RESET}")
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(f"{GRAY}  Powered by Google Gemini  |  Adaptive CEFR Teaching{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}\n")


def print_level_badge(level: str, levels: dict):
    info = levels.get(level, {})
    print(f"  {BOLD}{GREEN}Level: {level} — {info.get('name', '')}  {RESET}")
    print(f"  {GRAY}{info.get('description', '')}{RESET}\n")


def print_tutor(text: str):
    """Print the tutor's response with formatting."""
    print(f"\n{BOLD}{CYAN}🎓 Türkçe Hoca:{RESET}")
    # Handle bold markdown (**text**)
    import re
    formatted = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}\\1{RESET}', text)
    # Wrap at ~72 chars
    lines = formatted.split('\n')
    for line in lines:
        print(f"  {line}")
    print()


def print_prompt():
    """Print the student input prompt."""
    print(f"{BOLD}{YELLOW}📝 You:{RESET} ", end="", flush=True)


def print_separator():
    print(f"\n{GRAY}{'─' * 60}{RESET}")


def print_quick_help():
    print(f"\n{GRAY}Commands: /level, /topic, /exercise, /progress, /help, /quit{RESET}")


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_help():
    print(f"""
{BOLD}Available Commands:{RESET}
  {CYAN}/level{RESET}      — Change your CEFR level (A1, A2, B1, B2, C1, C2)
  {CYAN}/topic{RESET}      — Choose a specific topic to study
  {CYAN}/exercise{RESET}   — Get a practice exercise
  {CYAN}/vocab{RESET}      — Vocabulary flashcard drill
  {CYAN}/progress{RESET}   — Show your session progress
  {CYAN}/examples{RESET}   — Show grammar examples for current topic
  {CYAN}/quit / /exit{RESET} — End the session
  {CYAN}/help{RESET}       — Show this help menu
""")


def cmd_progress(session: TutorSession):
    print(f"""
{BOLD}📊 Session Progress:{RESET}
  Level:     {BOLD}{GREEN}{session.cefr_level}{RESET}
  Exchanges: {session.exchange_count}
  Correct:   {BOLD}{GREEN}{session.correct_answers}{RESET}
  Topics:    {', '.join(session.topics_covered) or 'none yet'}
  Duration:  started {session.started_at[:19]}
""")


def cmd_level_change(session: TutorSession, levels: dict) -> str:
    print(f"\n{BOLD}Available levels:{RESET}")
    for lvl, info in levels.items():
        marker = " ◀" if lvl == session.cefr_level else ""
        print(f"  {CYAN}{lvl}{RESET} — {info['name']}: {GRAY}{info['description'][:60]}...{RESET}{marker}")
    print_prompt()
    new_level = input().strip().upper()
    if new_level in levels:
        session.cefr_level = new_level
        print(f"\n  {GREEN}✅ Level changed to {new_level} — {levels[new_level]['name']}{RESET}")
        return new_level
    else:
        print(f"  {RED}Invalid level. Keeping {session.cefr_level}.{RESET}")
        return session.cefr_level


# ─── Exercises ────────────────────────────────────────────────────────────────

async def run_exercise(session: TutorSession, knowledge: List[Dict]) -> None:
    """Generate and run a single practice exercise."""
    from config import build_prompt

    exercise_prompt = build_prompt(
        question=f"Generate ONE practice exercise for a {session.cefr_level} Turkish student. "
                 "Pick the most useful exercise type for this level. "
                 "Present the question clearly, then wait for the student to answer. "
                 "Do NOT provide the answer yet.",
        knowledge_base=knowledge,
        cefr_level=session.cefr_level,
        conversation_history=session.history,
    )

    print(f"\n{BOLD}{BLUE}📝 Practice Exercise:{RESET}")
    response = await ask_gemini(exercise_prompt)
    print_tutor(response)
    session.add_exchange("assistant", response)

    # Get student answer
    print_prompt()
    answer = input().strip()
    if not answer or answer.startswith("/"):
        return

    session.add_exchange("user", answer)

    # Evaluate answer
    eval_prompt = build_prompt(
        question=f"The student answered: '{answer}'. Evaluate this answer to the exercise you just gave. "
                 "Was it correct? Explain why. Give the correct answer and a brief explanation.",
        knowledge_base=knowledge,
        cefr_level=session.cefr_level,
        conversation_history=session.history,
    )
    feedback = await ask_gemini(eval_prompt)
    print_tutor(feedback)
    session.add_exchange("assistant", feedback)

    # Simple heuristic: count as correct if response is positive
    if any(w in feedback.lower() for w in ["correct", "exactly", "well done", "bravo", "✅", "doğru", "mükemmel"]):
        session.correct_answers += 1


# ─── Vocab Flashcard ─────────────────────────────────────────────────────────

def run_vocab_drill(session: TutorSession, knowledge: List[Dict]) -> None:
    """Quick vocabulary flashcard drill from the knowledge base."""
    import random
    # Find vocabulary entries for this level
    vocab_entries = [e for e in knowledge
                     if e.get("category") == "vocabulary"
                     and e.get("level", "A1") == session.cefr_level]
    if not vocab_entries:
        vocab_entries = [e for e in knowledge if e.get("category") == "vocabulary"]

    if not vocab_entries:
        print(f"  {YELLOW}No vocabulary data found. Run: python dataset.py{RESET}")
        return

    entry = random.choice(vocab_entries)
    lines = [l for l in entry["content"].split("\n") if " = " in l]
    if not lines:
        return

    sample = random.sample(lines, min(5, len(lines)))
    print(f"\n{BOLD}🔤 Vocabulary Drill — {entry['topic']}{RESET}")
    print(f"{GRAY}Translate each word. Press Enter to reveal the answer.{RESET}\n")

    for pair in sample:
        tr_word, en_word = pair.split(" = ", 1)
        print(f"  {BOLD}{tr_word.strip()}{RESET} = ?  ", end="", flush=True)
        input()  # wait
        print(f"\033[1A\r  {BOLD}{CYAN}{tr_word.strip()}{RESET} = {GREEN}{en_word.strip()}{RESET}   ")


# ─── Main Chat Loop ───────────────────────────────────────────────────────────

async def chat_loop(session: TutorSession, knowledge: List[Dict]) -> None:
    """Main interactive tutoring loop."""
    from config import CEFR_LEVELS, build_prompt

    print_separator()
    print_level_badge(session.cefr_level, CEFR_LEVELS)

    # Opening message
    opening_prompt = build_prompt(
        question=f"Introduce yourself warmly to a new {session.cefr_level} student starting their first Turkish lesson. "
                 "Tell them what you'll cover today based on their level. Keep it to 3-4 sentences. Be encouraging!",
        knowledge_base=knowledge,
        cefr_level=session.cefr_level,
        conversation_history=[],
    )

    print(f"{GRAY}(Connecting to Türkçe Hoca...){RESET}", end="\r", flush=True)
    opening = await ask_gemini(opening_prompt)
    clear_line()
    print_tutor(opening)
    session.add_exchange("assistant", opening)
    print_quick_help()

    while True:
        print_separator()
        print_prompt()

        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        # Commands
        cmd = user_input.lower().strip()
        if cmd in ("/quit", "/exit", "/q"):
            print(f"\n  {GREEN}Güle güle! (Goodbye!) Great session! Hoşça kal! 🇹🇷{RESET}\n")
            break
        elif cmd == "/help":
            cmd_help()
            continue
        elif cmd == "/progress":
            cmd_progress(session)
            continue
        elif cmd == "/level":
            cmd_level_change(session, CEFR_LEVELS)
            continue
        elif cmd == "/exercise":
            await run_exercise(session, knowledge)
            continue
        elif cmd == "/vocab":
            run_vocab_drill(session, knowledge)
            continue

        # Regular question/message
        session.add_exchange("user", user_input)

        prompt = build_prompt(
            question=user_input,
            knowledge_base=knowledge,
            cefr_level=session.cefr_level,
            conversation_history=session.history[:-1],  # exclude most recent (already have it)
        )

        print(f"{GRAY}(thinking...){RESET}", end="\r", flush=True)
        response = await ask_gemini(prompt)
        clear_line()
        print_tutor(response)
        session.add_exchange("assistant", response)


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def run_tutor(args):
    from config import CEFR_LEVELS

    print_banner()

    # Check Gemini
    if not _init_gemini():
        print(f"  {RED}⚠️  GEMINI_API_KEY not set in .env{RESET}")
        print(f"  {GRAY}Add your key to .env: GEMINI_API_KEY=your_key_here{RESET}")
        print(f"  {GRAY}Get a free key at: https://aistudio.google.com{RESET}\n")
        sys.exit(1)

    # Load knowledge base (auto-build if missing)
    knowledge = load_knowledge()
    if not knowledge:
        print(f"  {RED}❌ Knowledge base is empty. Run: python dataset.py{RESET}")
        sys.exit(1)

    print(f"  {GREEN}✅ Gemini connected  |  📚 {len(knowledge)} knowledge topics loaded{RESET}\n")

    # Determine CEFR level
    cefr_level = args.level.upper() if args.level else None
    if not cefr_level:
        print(f"  {BOLD}What is your current Turkish level?{RESET}")
        for lvl, info in CEFR_LEVELS.items():
            print(f"    {CYAN}{lvl}{RESET} — {info['name']}: {GRAY}{info['description'][:55]}...{RESET}")
        print(f"\n  Enter level (A1/A2/B1/B2/C1/C2) or press Enter for A1: ", end="")
        choice = input().strip().upper()
        cefr_level = choice if choice in CEFR_LEVELS else "A1"

    # Create session
    session = TutorSession(cefr_level=cefr_level)

    # If --exercise flag, jump straight to exercise
    if args.exercise:
        await run_exercise(session, knowledge)
    else:
        await chat_loop(session, knowledge)

    # Save session
    session.save()
    print(f"  {GRAY}Session saved. Total exchanges: {session.exchange_count}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="🇹🇷 Türkçe Hoca — AI Turkish Language Tutor")
    parser.add_argument("--level", type=str, choices=["A1", "A2", "B1", "B2", "C1", "C2"],
                        help="CEFR level to start at (default: ask on startup)")
    parser.add_argument("--topic", type=str, help="Specific topic to focus on")
    parser.add_argument("--exercise", action="store_true", help="Start immediately with an exercise")
    args = parser.parse_args()

    asyncio.run(run_tutor(args))


if __name__ == "__main__":
    main()
