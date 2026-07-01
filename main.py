#!/usr/bin/env python3
"""
main.py — Prompt Engineering Evaluator CLI

Usage:
  python main.py --task "Summarise a news article in one sentence" --cases test_cases.json
  python main.py --task "Debug this Python function" --cases my_cases.json --output out.json

Architecture:
  CLI input → validate → Generator Agent → 3 prompts
           → parallel Runner (3×N LLM calls)
           → parallel Evaluator (3×N scored)
           → Console table + JSON report

Security pre-run checklist (security-first-dev):
  ✓ All user inputs validated (task length/pattern, file path, test case schema)
  ✓ API key validated at startup — never hardcoded
  ✓ Output path checked against CWD to prevent path traversal (reporter.py)
  ✓ No load()/loads() of LLM output (CVE-2025-68664 / CVE-2026-44843 mitigated)
  ✓ No load_prompt() from user-controlled paths (CVE-2026-34070 mitigated)
  ✓ No subprocess / shell=True anywhere
  ✓ Error messages to console are generic; full tracebacks go to log file only
  ✓ langchain-core pinned to >=1.4.8 (all known CVEs patched)

Written for: langchain==1.3.11, langchain-core==1.4.8, langchain-groq==1.1.2
             Python >=3.10
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from agents.evaluator import evaluate_output
from agents.generator import generate_prompts
from models import EvaluationScore
from reporter import (
    aggregate_scores,
    print_detail_table,
    print_prompts,
    print_summary_table,
    print_winner_analysis,
    save_json_report,
)
from runner import run_all_parallel

# ─── Logging Setup ────────────────────────────────────────────────────────────
# Logs go to file; console only gets user-friendly messages.
# This prevents internal details (stack traces, API errors) from leaking.

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("evaluator.log")],
)
logger = logging.getLogger(__name__)
console = Console()

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_TASK_LEN = 500
MIN_TASK_LEN = 10
MAX_TEST_CASES = 20
MAX_INPUT_LEN = 2000

# Patterns that indicate genuine prompt-injection attempts in the task string.
# Deliberately narrow: require phrase boundaries and strong injection signals
# to avoid false positives on legitimate tasks like "design a database system:
# explain normalization" or "DAN is a character name".
_INJECTION_PATTERNS = re.compile(
    r"(?:"
    r"ignore\s+(all\s+)?previous\s+instructions"   # classic instruction override
    r"|disregard\s+all\s+(previous\s+)?instructions"
    r"|you\s+are\s+now\s+(?:DAN|an?\s+AI\s+without)"  # DAN jailbreak phrasing
    r"|pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(restrictions|rules|guidelines)"
    r"|\bdo\s+anything\s+now\b"                    # "DAN" expansion
    r")",
    re.IGNORECASE,
)

# ─── Input Validation ─────────────────────────────────────────────────────────

def validate_task(raw: str) -> str:
    """Sanitise and validate the task description from CLI.

    Checks:
      - Non-empty, within length bounds.
      - No obvious prompt-injection patterns.

    Does NOT strip HTML or encode characters — this is a CLI tool sending
    text to an LLM, not rendering it in a browser.
    """
    task = raw.strip()

    if len(task) < MIN_TASK_LEN:
        raise ValueError(
            f"Task description is too short (minimum {MIN_TASK_LEN} characters)."
        )
    if len(task) > MAX_TASK_LEN:
        raise ValueError(
            f"Task description is too long ({len(task)} chars, max {MAX_TASK_LEN})."
        )
    if _INJECTION_PATTERNS.search(task):
        raise ValueError(
            "Task description contains a disallowed pattern. "
            "Please describe the task without meta-instructions."
        )

    return task


def load_test_cases(path_str: str) -> list[dict]:
    """Load and validate test cases from a JSON file.

    Validates:
      - File exists and is within a safe path (no directory traversal).
      - JSON is a list of dicts with 'id' (str) and 'input' (str).
      - Count and individual input length within bounds.
    """
    # Resolve to absolute path and ensure it doesn't escape CWD.
    p = Path(path_str).resolve()
    cwd = Path.cwd().resolve()
    try:
        p.relative_to(cwd)
    except ValueError:
        raise ValueError(
            f"Test cases path '{path_str}' is outside the working directory."
        )

    if not p.exists():
        raise FileNotFoundError(f"Test cases file not found: {p}")
    if not p.is_file():
        raise ValueError(f"'{p}' is not a file.")
    if p.suffix.lower() != ".json":
        raise ValueError("Test cases file must have a .json extension.")

    # Guard against reading enormous files into memory.
    max_bytes = 1 * 1024 * 1024  # 1 MB — more than enough for 20 test cases
    if p.stat().st_size > max_bytes:
        raise ValueError(
            f"Test cases file is too large ({p.stat().st_size // 1024} KB). "
            f"Maximum allowed: {max_bytes // 1024} KB."
        )

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Test cases file is not valid JSON: {e}")

    if not isinstance(raw, list):
        raise ValueError("Test cases file must contain a JSON array at the top level.")
    if len(raw) == 0:
        raise ValueError("Test cases file contains no test cases.")
    if len(raw) > MAX_TEST_CASES:
        raise ValueError(
            f"Too many test cases ({len(raw)}). Maximum allowed: {MAX_TEST_CASES}."
        )

    seen_ids: set[str] = set()
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            raise ValueError(f"Test case {i} must be a JSON object, got {type(tc).__name__}.")

        for key in ("id", "input"):
            if key not in tc:
                raise ValueError(f"Test case {i} is missing required key: '{key}'.")
            if not isinstance(tc[key], str):
                raise ValueError(f"Test case {i}: '{key}' must be a string.")
            if not tc[key].strip():
                raise ValueError(f"Test case {i}: '{key}' must not be empty.")

        if len(tc["input"]) > MAX_INPUT_LEN:
            raise ValueError(
                f"Test case '{tc['id']}': 'input' exceeds {MAX_INPUT_LEN} chars "
                f"({len(tc['input'])} chars)."
            )

        tc_id = tc["id"].strip()
        if tc_id in seen_ids:
            raise ValueError(f"Duplicate test case id: '{tc_id}'.")
        seen_ids.add(tc_id)

    return raw


def get_api_key() -> str:
    """Read and do a basic format-check on GROQ_API_KEY.

    Never logs or prints the key itself.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your Groq API key (get one free at console.groq.com)\n"
            "  3. Re-run this command."
        )
    # Groq API keys begin with "gsk_" and are >20 chars.
    if not key.startswith("gsk_") or len(key) < 20:
        raise EnvironmentError(
            "GROQ_API_KEY appears malformed (expected to start with 'gsk_'). "
            "Check your .env file."
        )
    return key


# ─── Async Orchestration ──────────────────────────────────────────────────────

async def _run_evaluations_parallel(
    results,
    task: str,
    api_key: str,
    max_concurrent: int,
) -> list[EvaluationScore]:
    """Run all evaluations in parallel under a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval_one(r):
        async with semaphore:
            return await evaluate_output(r, task, api_key)

    raw = await asyncio.gather(
        *[_eval_one(r) for r in results],
        return_exceptions=True,
    )

    scores: list[EvaluationScore] = []
    for item in raw:
        if isinstance(item, Exception):
            logger.warning("Evaluation failed: %s", item)
            console.print(f"[yellow]⚠ One evaluation failed (see evaluator.log)[/yellow]")
        else:
            scores.append(item)

    return scores


async def orchestrate(
    task: str,
    test_cases: list[dict],
    output_path: str,
    api_key: str,
) -> None:
    """Top-level async orchestration: generate → run → evaluate → report."""
    max_concurrent = int(os.environ.get("MAX_CONCURRENT", "6"))

    total_runs = 3 * len(test_cases)
    console.print(
        f"\n[bold]Prompt Engineering Evaluator[/bold]\n"
        f"  Task       : {task}\n"
        f"  Test cases : {len(test_cases)}\n"
        f"  LLM calls  : 1 (generate) + {total_runs} (run) + {total_runs} (evaluate) "
        f"= {1 + 2 * total_runs} total\n"
    )

    # ── Step 1: Generate 3 prompts ──────────────────────────────────────────
    console.print("[bold cyan][1/3][/bold cyan] Generating prompt strategies...")
    try:
        prompts = await generate_prompts(task, api_key)
    except Exception as exc:
        logger.error("Generator failed: %s", exc, exc_info=True)
        console.print("[red]Generator failed.[/red] Check evaluator.log for details.")
        sys.exit(1)

    print_prompts(prompts)

    # ── Step 2: Run all (prompt × test_case) in parallel ───────────────────
    console.print(f"[bold cyan][2/3][/bold cyan] Running {total_runs} parallel LLM calls...")
    try:
        results = await run_all_parallel(prompts, test_cases, api_key)
    except Exception as exc:
        logger.error("Runner failed: %s", exc, exc_info=True)
        console.print("[red]Runner failed.[/red] Check evaluator.log for details.")
        sys.exit(1)

    if not results:
        console.print("[red]No outputs collected. Cannot evaluate.[/red]")
        sys.exit(1)

    console.print(f"[green]✓[/green] {len(results)}/{total_runs} outputs collected.\n")

    # ── Step 3: Evaluate all outputs in parallel ────────────────────────────
    console.print(
        f"[bold cyan][3/3][/bold cyan] Evaluating {len(results)} outputs "
        f"(temperature=0, max_concurrent={max_concurrent})..."
    )
    scores = await _run_evaluations_parallel(results, task, api_key, max_concurrent)

    if not scores:
        console.print("[red]No evaluations succeeded. Cannot produce a report.[/red]")
        sys.exit(1)

    console.print(f"[green]✓[/green] {len(scores)} evaluations complete.\n")

    # ── Step 4: Report ──────────────────────────────────────────────────────
    summaries = aggregate_scores(scores)
    print_summary_table(task, summaries)
    print_detail_table(scores)
    print_winner_analysis(summaries)

    try:
        save_json_report(task, prompts, results, scores, summaries, output_path)
    except (PermissionError, OSError) as exc:
        console.print(f"[yellow]Could not save JSON report: {exc}[/yellow]")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt_evaluator",
        description=(
            "Auto-generates 3 prompt strategies for a task, runs them against test cases, "
            "and scores each on Accuracy, Brevity, and Helpfulness."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --task "Summarise a news article in one sentence" --cases test_cases.json
  python main.py --task "Debug this Python function" --cases debug_cases.json --output debug_results.json
  python main.py --task "Explain quantum entanglement to a 10-year-old" --cases explain_cases.json
        """,
    )
    parser.add_argument(
        "--task",
        required=True,
        metavar="DESCRIPTION",
        help=f"What the LLM should do. ({MIN_TASK_LEN}–{MAX_TASK_LEN} chars)",
    )
    parser.add_argument(
        "--cases",
        default="test_cases.json",
        metavar="FILE",
        help="Path to test cases JSON (default: test_cases.json)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        metavar="FILE",
        help="Where to write the JSON report (default: results.json)",
    )
    return parser


def main() -> None:
    load_dotenv()  # Load .env before anything else

    parser = _build_parser()
    args = parser.parse_args()

    # Validate all inputs before any network call.
    try:
        task = validate_task(args.task)
    except ValueError as e:
        console.print(f"[red]--task error:[/red] {e}")
        sys.exit(1)

    try:
        test_cases = load_test_cases(args.cases)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]--cases error:[/red] {e}")
        sys.exit(1)

    try:
        api_key = get_api_key()
    except EnvironmentError as e:
        console.print(f"[red]Config error:[/red]\n{e}")
        sys.exit(1)

    asyncio.run(orchestrate(task, test_cases, args.output, api_key))


if __name__ == "__main__":
    main()
