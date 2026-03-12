"""
Turkish Agent Tutor — Autoresearch Swarm
Manages parallel experiments to improve tutor teaching quality.

Each branch experiments with a different config.py teaching strategy.
The evaluate.py score determines the winner.

Usage:
    python swarm.py --spawn 3              # Launch 3 parallel experiment branches
    python swarm.py --leaderboard          # Show best scores across branches
    python swarm.py --adopt experiment/agent-1 --confirm  # Promote best config
    python swarm.py --status               # Show all experiment branches
"""

import os
import json
import re
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Optional

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")

# Teaching strategy directions agents can explore
RESEARCH_DIRECTIONS = [
    {
        "name": "socratic-method",
        "hint": "Tune the tutor to use more Socratic questioning: ask guiding questions "
                "before giving answers. Modify STRATEGY['error_correction_style'] and "
                "FEW_SHOT_EXAMPLES to reflect this approach. Measure if accuracy scores improve.",
    },
    {
        "name": "mnemonic-heavy",
        "hint": "Add rich mnemonics and memory stories for every new vocabulary item. "
                "Update use_mnemonics=True and add mnemonic examples to FEW_SHOT_EXAMPLES. "
                "Turkish vowel harmony is a great candidate for visual/story-based mnemonics.",
    },
    {
        "name": "cefr-strict-calibration",
        "hint": "Strictly enforce CEFR level by filtering knowledge retrieval to only "
                "show same-level or one-level-up content. Update retrieve_context() to "
                "hard-filter by level. Measure if level_calibration score improves.",
    },
    {
        "name": "cultural-immersion",
        "hint": "Add rich Turkish cultural context to every lesson: Turkish proverbs, "
                "music references, idioms, and real-world usage examples. "
                "Extend GRAMMAR_RULES in dataset.py with cultural notes.",
    },
    {
        "name": "spaced-repetition",
        "hint": "Implement active spaced repetition: after every 5 exchanges, the tutor "
                "should bring back vocabulary from earlier in the session for review. "
                "Update spaced_repetition_interval and build_prompt() to inject review.",
    },
    {
        "name": "error-first-teaching",
        "hint": "Make the tutor present common learner errors first, then teach the correct form. "
                "Research shows error analysis improves retention. Add a 'common_mistakes' "
                "section to each grammar rule and update FEW_SHOT_EXAMPLES to use this pattern.",
    },
    {
        "name": "comprehensible-input-plus-1",
        "hint": "Implement Krashen's i+1 hypothesis: always present content one step above "
                "the learner's confirmed level. Tune CEFR_LEVELS thresholds and context "
                "retrieval to systematically include some content from the next level up.",
    },
    {
        "name": "morpheme-first",
        "hint": "Redesign grammar explanations to always start from the morpheme level. "
                "Every Turkish word should be broken into its constituent suffixes before "
                "the meaning is explained. Update GRAMMAR_RULES and SYSTEM_PROMPT.",
    },
]


# ─── Git Operations ───────────────────────────────────────────────────────────

def git(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        capture_output=True, text=True,
        cwd=REPO_DIR, check=check
    )


def get_current_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def get_default_branch() -> str:
    """Resolve the repository default branch with sensible fallbacks."""
    origin_head = git(["symbolic-ref", "refs/remotes/origin/HEAD"], check=False)
    if origin_head.returncode == 0:
        ref = origin_head.stdout.strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref.replace("refs/remotes/origin/", "", 1)

    for candidate in ("main", "master"):
        if branch_exists(candidate):
            return candidate

    return get_current_branch()


def has_uncommitted_changes() -> bool:
    """Return True when the working tree has tracked or untracked changes."""
    status = git(["status", "--porcelain"], check=False)
    return bool(status.stdout.strip())


def list_experiment_branches() -> List[str]:
    result = git(["branch", "--list", "experiment/*"])
    return [b.strip().lstrip("* ") for b in result.stdout.strip().split("\n") if b.strip()]


def branch_exists(name: str) -> bool:
    return bool(git(["branch", "--list", name]).stdout.strip())


def get_branch_score(branch: str) -> Optional[float]:
    """Get the best composite score from a branch's commit messages."""
    result = git(["log", branch, "--oneline", "-20", "--format=%s"], check=False)
    if result.returncode != 0:
        return None
    best = 0.0
    for line in result.stdout.strip().split("\n"):
        if "score=" in line:
            try:
                score_str = line.split("score=")[1].split()[0].strip("()+")
                score = float(score_str)
                best = max(best, score)
            except (ValueError, IndexError):
                continue
    return best if best > 0 else None


# ─── Swarm Operations ─────────────────────────────────────────────────────────

def spawn_branches(count: int, dry_run: bool = False) -> List[Dict]:
    """Create N experiment branches, each with a different research direction."""
    original_branch = get_current_branch()
    base_branch = get_default_branch()
    existing = list_experiment_branches()

    available = RESEARCH_DIRECTIONS
    spawned = []
    agent_num = len(existing) + 1

    for i in range(count):
        direction = available[i % len(available)]
        branch_name = f"experiment/agent-{agent_num + i}"

        if branch_exists(branch_name):
            print(f"  ⚠️  Branch {branch_name} already exists, skipping")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would create: {branch_name} → {direction['name']}")
            spawned.append({"branch": branch_name, "direction": direction["name"]})
            continue

        git(["checkout", base_branch], check=False)
        git(["checkout", "-b", branch_name])

        direction_file = os.path.join(REPO_DIR, ".direction")
        with open(direction_file, "w", encoding="utf-8") as f:
            json.dump({
                "agent_id": f"agent-{agent_num + i}",
                "branch": branch_name,
                "direction": direction["name"],
                "hint": direction["hint"],
                "created_at": datetime.now().isoformat(),
            }, f, indent=2)

        git(["add", "-f", ".direction"])
        git(["commit", "-m",
             f"chore: assign direction [{direction['name']}] to {branch_name}"])

        print(f"  ✅ Created {branch_name} → {direction['name']}")
        spawned.append({"branch": branch_name, "direction": direction["name"]})

    if not dry_run:
        git(["checkout", original_branch], check=False)

    return spawned


def show_status():
    branches = list_experiment_branches()
    current = get_current_branch()
    base_branch = get_default_branch()

    print(f"\n{'═' * 68}")
    print(f"  🐝 Turkish Tutor Swarm — {len(branches)} experiment branches")
    print(f"  Current branch: {current}")
    print(f"{'═' * 68}\n")

    if not branches:
        print("  No experiment branches. Run: python swarm.py --spawn 3\n")
        return

    for branch in sorted(branches):
        direction = "?"
        result = git(["show", f"{branch}:.direction"], check=False)
        if result.returncode == 0:
            try:
                info = json.loads(result.stdout)
                direction = info.get("direction", "?")
            except json.JSONDecodeError:
                pass

        score = get_branch_score(branch)
        score_str = f"score={score:.4f}" if score else "no results"

        result = git(["rev-list", "--count", f"{base_branch}..{branch}"], check=False)
        commits = result.stdout.strip() if result.returncode == 0 else "?"

        active = " ← YOU" if branch == current else ""
        print(f"  {branch:35s}  [{direction:25s}]  {score_str:12s}  {commits} commits{active}")

    print(f"\n{'═' * 68}\n")


def show_leaderboard():
    branches = list_experiment_branches()

    print(f"\n{'═' * 68}")
    print("  🏆 Turkish Tutor Leaderboard")
    print(f"{'═' * 68}\n")

    entries = []
    for branch in branches:
        score = get_branch_score(branch)
        direction = "?"
        result = git(["show", f"{branch}:.direction"], check=False)
        if result.returncode == 0:
            try:
                info = json.loads(result.stdout)
                direction = info.get("direction", "?")
            except json.JSONDecodeError:
                pass
        if score:
            entries.append({"source": branch, "score": score, "direction": direction})

    # Also check local eval results
    results_dir = os.path.join(DATA_DIR, "eval_results")
    if os.path.isdir(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    score = data.get("scores", {}).get("composite", 0)
                    if score > 0:
                        entries.append({
                            "source": fname.replace("eval_", "").replace(".json", ""),
                            "score": score,
                            "direction": "local",
                        })
                except (OSError, json.JSONDecodeError, ValueError):
                    pass

    entries.sort(key=lambda x: -x["score"])
    seen = set()
    unique = []
    for e in entries:
        key = (e["source"], round(e["score"], 4))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    if not unique:
        print("  No scores yet. Run: python evaluate.py")
    else:
        for i, e in enumerate(unique[:15]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1:2d}"
            bar = "█" * int(e["score"] * 20)
            print(f"  {medal}  score={e['score']:.4f}  {bar}  [{e['direction']:25s}]  {e['source']}")

    print(f"\n{'═' * 68}\n")


def adopt_branch(branch: str, confirm: bool = False):
    """Copy config.py from the specified branch to default branch.

    Security: Only experiment/* branches with safe names are accepted.
    --confirm flag required to prevent accidental overwrites.
    """
    if not re.match(r'^experiment/[a-zA-Z0-9_\-]+$', branch):
        print(f"  ❌ Invalid branch name '{branch}'. Only 'experiment/<safe-name>' accepted.")
        return

    if has_uncommitted_changes():
        print("  ❌ Working tree has uncommitted changes.")
        print("     Commit or stash them before adopting a branch to avoid accidental overwrite.")
        return

    if not branch_exists(branch):
        print(f"  ❌ Branch '{branch}' does not exist.")
        return

    result = git(["show", f"{branch}:config.py"], check=False)
    if result.returncode != 0:
        print(f"  ❌ Could not read config.py from {branch}")
        return

    score = get_branch_score(branch)
    score_str = f"score={score:.4f}" if score else "unknown score"

    base_branch = get_default_branch()

    if not confirm:
        print(f"  ⚠️  This will overwrite config.py on {base_branch} with code from {branch} ({score_str}).")
        print(f"     Review first: git show {branch}:config.py")
        print(f"     To proceed: python swarm.py --adopt {branch} --confirm")
        return

    original = get_current_branch()
    git(["checkout", base_branch])

    config_path = os.path.join(REPO_DIR, "config.py")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    git(["add", "config.py"])
    git(["commit", "-m",
         f"adopt: {branch} ({score_str})\n\nCherry-picked config.py from {branch}"])

    print(f"  ✅ Adopted config.py from {branch} ({score_str}) into {base_branch}")

    if original != base_branch:
        git(["checkout", original], check=False)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="🐝 Turkish Tutor Swarm — Multi-Agent Strategy Optimizer")
    parser.add_argument("--spawn", type=int, metavar="N", help="Create N experiment branches")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--adopt", type=str, metavar="BRANCH")
    parser.add_argument("--confirm", action="store_true",
                        help="Required to confirm --adopt (prevents accidental config overwrite)")
    parser.add_argument("--directions", action="store_true")
    args = parser.parse_args()

    if args.spawn:
        print(f"\n🐝 Spawning {args.spawn} experiment branches...\n")
        spawned = spawn_branches(args.spawn, dry_run=args.dry_run)
        print(f"\n  Spawned {len(spawned)} branches. Checkout a branch and modify config.py,")
        print("  then run: python evaluate.py  to score it.\n")
    elif args.status:
        show_status()
    elif args.leaderboard:
        show_leaderboard()
    elif args.adopt:
        adopt_branch(args.adopt, confirm=args.confirm)
    elif args.directions:
        print(f"\n{'═' * 58}")
        print("  📋 Available Research Directions")
        print(f"{'═' * 58}\n")
        for d in RESEARCH_DIRECTIONS:
            print(f"  → {d['name']}")
            print(f"    {d['hint'][:90]}...\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
