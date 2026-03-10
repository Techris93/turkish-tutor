"""
Turkish Agent Tutor — Evaluator
Scores the tutor config.py against the test Q&A dataset.
Used by the autoresearch swarm to compare experiment branches.

Metrics:
  - answer_accuracy:    semantic similarity of AI answers vs gold standard
  - pedagogy_score:     tone, encouragement, explanation quality
  - turkish_correctness: grammatical quality of Turkish in responses
  - level_calibration:  appropriate difficulty for the CEFR level

Usage:
    python evaluate.py               # Run full evaluation
    python evaluate.py --verbose     # Show per-question results
    python evaluate.py --level A2    # Evaluate only A2 questions
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_FILE   = os.path.join(DATA_DIR, "test_qa.json")
RESULTS_DIR = os.path.join(DATA_DIR, "eval_results")

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


# Max retries for 429 rate-limit errors
_MAX_RETRIES = 4
_BASE_DELAY  = 5   # seconds — initial backoff


async def _generate(prompt: str, max_tokens: int = 400) -> str:
    """Call Gemini with retry + exponential backoff for 429 rate limits."""
    if not GEMINI_AVAILABLE:
        return ""

    last_error = None
    for attempt in range(_MAX_RETRIES):
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
                            temperature=0.2,
                            max_output_tokens=max_tokens,
                        )
                    )
                )
            except (ImportError, AttributeError):
                response = await loop.run_in_executor(
                    None,
                    lambda: _client.generate_content(prompt)
                )
            return response.text.strip()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate limit (429) or transient server errors (5xx)
            if "429" in error_str or "quota" in error_str or "resource" in error_str or "500" in error_str:
                wait = _BASE_DELAY * (2 ** attempt)
                print(f"\n  ⏳ Rate limited (attempt {attempt+1}/{_MAX_RETRIES}), waiting {wait}s...", end="", flush=True)
                await asyncio.sleep(wait)
            else:
                return f"[error: {e}]"  # Non-retryable error

    return f"[error after {_MAX_RETRIES} retries: {last_error}]"


# ─── Load knowledge ───────────────────────────────────────────────────────────

def load_knowledge() -> List[Dict]:
    knowledge_file = os.path.join(DATA_DIR, "knowledge.json")
    if not os.path.exists(knowledge_file):
        return []
    with open(knowledge_file, encoding="utf-8") as f:
        return json.load(f).get("knowledge_base", [])


# ─── Evaluation Functions ─────────────────────────────────────────────────────

async def get_tutor_answer(question: str, knowledge: List[Dict], level: str) -> str:
    """Get the tutor's answer to a test question."""
    from config import build_prompt
    prompt = build_prompt(
        question=question,
        knowledge_base=knowledge,
        cefr_level=level,
        conversation_history=[],
    )
    return await _generate(prompt, max_tokens=400)


async def score_accuracy(tutor_answer: str, gold_answer: str, question: str) -> float:
    """Score how accurate the tutor's answer is vs the gold standard (0.0–1.0)."""
    prompt = f"""You are a Turkish language teaching evaluator.

Question: {question}
Gold standard answer: {gold_answer}
Tutor's answer: {tutor_answer}

Score the FACTUAL ACCURACY of the tutor's answer on a scale from 0.0 to 1.0:
  1.0 = Completely correct and covers all key facts
  0.7 = Mostly correct with minor omissions
  0.5 = Partially correct
  0.3 = Contains a significant error
  0.0 = Wrong or contradicts the gold answer

Respond with ONLY a number between 0.0 and 1.0. No explanation."""
    result = await _generate(prompt, max_tokens=5)
    try:
        return min(1.0, max(0.0, float(result.strip())))
    except ValueError:
        return 0.5


async def score_pedagogy(tutor_answer: str, level: str) -> float:
    """Score pedagogical quality of the response (0.0–1.0)."""
    prompt = f"""Evaluate the pedagogical quality of this Turkish language tutor response for a {level} student.

Tutor response:
{tutor_answer}

Score on a scale from 0.0 to 1.0 considering:
  - Clarity of explanation
  - Use of examples
  - Encouraging and appropriate tone for {level} level
  - Does it include a practice element or follow-up question?
  - Appropriate length (not too long, not too short)

Respond with ONLY a number between 0.0 and 1.0."""
    result = await _generate(prompt, max_tokens=5)
    try:
        return min(1.0, max(0.0, float(result.strip())))
    except ValueError:
        return 0.5


async def score_turkish_correctness(tutor_answer: str) -> float:
    """Score whether Turkish words/phrases in the response are grammatically correct (0.0–1.0)."""
    prompt = f"""Evaluate the Turkish language correctness in this tutor response.
Check if any Turkish words, phrases, or examples are grammatically correct.

Response: {tutor_answer}

Score 0.0 to 1.0:
  1.0 = All Turkish is grammatically correct
  0.7 = Minor errors or typos
  0.5 = One notable error
  0.2 = Multiple errors
  0.0 = Mostly incorrect Turkish

If there is no Turkish content in the response, score 0.8 (neutral).
Respond with ONLY a number between 0.0 and 1.0."""
    result = await _generate(prompt, max_tokens=5)
    try:
        return min(1.0, max(0.0, float(result.strip())))
    except ValueError:
        return 0.8


async def evaluate_question(
    qa: Dict, knowledge: List[Dict], verbose: bool = False
) -> Dict[str, Any]:
    """Full evaluation of one test question."""
    question = qa["question"]
    gold = qa["expected_answer"]
    level = qa.get("level", "A2")

    tutor_answer = await get_tutor_answer(question, knowledge, level)

    # Score dimensions sequentially (not parallel) to avoid burst rate limits
    acc = await score_accuracy(tutor_answer, gold, question)
    await asyncio.sleep(1)  # brief pause between scoring calls
    ped = await score_pedagogy(tutor_answer, level)
    await asyncio.sleep(1)
    tr  = await score_turkish_correctness(tutor_answer)

    # Composite score: weighted average
    composite = (acc * 0.50) + (ped * 0.30) + (tr * 0.20)

    result = {
        "question": question,
        "level": level,
        "category": qa.get("category", "?"),
        "difficulty": qa.get("difficulty", "?"),
        "gold_answer": gold,
        "tutor_answer": tutor_answer,
        "accuracy": round(acc, 3),
        "pedagogy": round(ped, 3),
        "turkish_correctness": round(tr, 3),
        "composite": round(composite, 3),
    }

    if verbose:
        print(f"\n  Q: {question[:60]}...")
        print(f"     Tutor: {tutor_answer[:80]}...")
        print(f"     Accuracy={acc:.2f}  Pedagogy={ped:.2f}  Turkish={tr:.2f}  → {composite:.3f}")

    return result


async def run_evaluation(
    level_filter: Optional[str] = None,
    verbose: bool = False,
    max_questions: int = 20,
) -> Dict[str, Any]:
    """Run the full evaluation suite."""
    if not os.path.exists(TEST_FILE):
        print("❌ No test dataset. Run: python dataset.py")
        sys.exit(1)

    with open(TEST_FILE, encoding="utf-8") as f:
        test_data = json.load(f)

    questions = test_data.get("test_pairs", [])
    if level_filter:
        questions = [q for q in questions if q.get("level") == level_filter]
    questions = questions[:max_questions]

    knowledge = load_knowledge()

    print(f"\n📊 Evaluating {len(questions)} questions", end="", flush=True)
    if level_filter:
        print(f" (level: {level_filter})", end="")
    print("...\n")

    # Delay between questions to avoid rate-limit (429)
    DELAY_BETWEEN_QUESTIONS = 3  # seconds

    results = []
    for i, qa in enumerate(questions):
        print(f"  [{i+1:2d}/{len(questions)}] {qa['question'][:50]}...", end="\r", flush=True)
        result = await evaluate_question(qa, knowledge, verbose=verbose)
        results.append(result)
        # Pace the calls — the free tier allows ~15 RPM
        if i < len(questions) - 1:
            await asyncio.sleep(DELAY_BETWEEN_QUESTIONS)

    print(" " * 70, end="\r")  # clear the last line

    # Aggregate scores
    if not results:
        return {}

    avg_accuracy  = sum(r["accuracy"]  for r in results) / len(results)
    avg_pedagogy  = sum(r["pedagogy"]  for r in results) / len(results)
    avg_turkish   = sum(r["turkish_correctness"] for r in results) / len(results)
    avg_composite = sum(r["composite"] for r in results) / len(results)

    # Per-category breakdown
    by_category: Dict[str, List[float]] = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r["composite"])
    category_scores = {cat: round(sum(scores) / len(scores), 3)
                       for cat, scores in by_category.items()}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "questions_evaluated": len(results),
        "level_filter": level_filter,
        "scores": {
            "accuracy":          round(avg_accuracy, 4),
            "pedagogy":          round(avg_pedagogy, 4),
            "turkish_correctness": round(avg_turkish, 4),
            "composite":         round(avg_composite, 4),
        },
        "by_category": category_scores,
        "results": results,
    }

    return summary


def save_results(summary: Dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def print_results(summary: Dict):
    scores = summary.get("scores", {})
    composite = scores.get("composite", 0)

    print(f"\n{'═' * 55}")
    print(f"  🇹🇷 Turkish Tutor — Evaluation Results")
    print(f"{'═' * 55}")
    print(f"  Questions: {summary['questions_evaluated']}")
    print(f"\n  Composite Score:    {composite:.4f}  ({'🟢 Good' if composite >= 0.7 else '🟡 OK' if composite >= 0.5 else '🔴 Needs work'})")
    print(f"  Accuracy:           {scores.get('accuracy', 0):.4f}")
    print(f"  Pedagogy:           {scores.get('pedagogy', 0):.4f}")
    print(f"  Turkish Correctness:{scores.get('turkish_correctness', 0):.4f}")
    print(f"\n  By Category:")
    for cat, score in sorted(summary.get("by_category", {}).items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"    {cat:25s} {score:.3f}  {bar}")
    print(f"{'═' * 55}\n")


async def main_async(args):
    if not _init_gemini():
        print("❌ GEMINI_API_KEY not set in .env")
        sys.exit(1)

    summary = await run_evaluation(
        level_filter=args.level,
        verbose=args.verbose,
        max_questions=args.max_questions,
    )

    if not summary:
        print("No results to display.")
        return

    print_results(summary)
    path = save_results(summary)
    composite = summary["scores"]["composite"]
    print(f"  Results saved: {path}")
    print(f"\n  Composite F1-equivalent: {composite:.4f}")
    print(f"  (This score is used by swarm.py to rank experiment branches)\n")


def main():
    parser = argparse.ArgumentParser(description="🇹🇷 Turkish Tutor — Evaluator")
    parser.add_argument("--level", choices=["A1", "A2", "B1", "B2", "C1", "C2"],
                        help="Evaluate only questions for this CEFR level")
    parser.add_argument("--verbose", action="store_true", help="Show per-question scores")
    parser.add_argument("--max-questions", type=int, default=20,
                        help="Max questions to evaluate (default: 20)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
