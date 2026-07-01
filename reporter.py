"""
reporter.py — Console + JSON Report Generator

Aggregates EvaluationScore objects → pretty Rich tables + structured JSON.

No user-supplied data is passed to any format string without explicit typing,
preventing accidental injection into the report format.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from models import EvaluationScore, GeneratedPrompts, PromptResult, PromptSummary

logger = logging.getLogger(__name__)
console = Console()

STRATEGY_NAMES = {
    "A": "Zero-shot",
    "B": "Chain-of-thought",
    "C": "Few-shot",
}

SCORE_COLOURS = {
    5: "bold green",
    4: "green",
    3: "yellow",
    2: "red",
    1: "bold red",
}


def _colour_score(score: int | float) -> Text:
    rounded = round(float(score))
    colour = SCORE_COLOURS.get(rounded, "white")
    return Text(str(score), style=colour)


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_scores(scores: list[EvaluationScore]) -> list[PromptSummary]:
    """Average scores per prompt label."""
    by_label: dict[str, list[EvaluationScore]] = defaultdict(list)
    for s in scores:
        by_label[s.prompt_label].append(s)

    summaries = []
    for label in sorted(by_label):
        group = by_label[label]
        n = len(group)
        summaries.append(
            PromptSummary(
                label=label,
                strategy=STRATEGY_NAMES.get(label, "Unknown"),
                avg_accuracy=round(sum(s.accuracy for s in group) / n, 2),
                avg_brevity=round(sum(s.brevity for s in group) / n, 2),
                avg_helpfulness=round(sum(s.helpfulness for s in group) / n, 2),
                avg_total=round(sum(s.total for s in group) / n, 2),
                n_evaluations=n,
            )
        )
    return summaries


# ─── Console Output ───────────────────────────────────────────────────────────

def print_prompts(prompts: GeneratedPrompts) -> None:
    """Display the 3 generated prompts in Rich panels."""
    console.print("\n[bold underline]Generated Prompts[/bold underline]\n")
    for label, strategy, text in [
        ("A", "Zero-shot",        prompts.prompt_a),
        ("B", "Chain-of-thought", prompts.prompt_b),
        ("C", "Few-shot",         prompts.prompt_c),
    ]:
        preview = text[:300] + ("…" if len(text) > 300 else "")
        console.print(
            Panel(
                preview,
                title=f"[bold]Prompt {label}[/bold] — {strategy}",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    console.print(f"[dim]Rationale: {prompts.rationale}[/dim]\n")


def print_summary_table(task: str, summaries: list[PromptSummary]) -> None:
    """Print the aggregated comparison table."""
    console.rule("[bold cyan]Evaluation Summary[/bold cyan]")
    console.print(f"[dim]Task:[/dim] {task}\n")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title="Prompt Strategy Comparison",
    )
    table.add_column("Prompt",        style="bold",  width=10)
    table.add_column("Strategy",      style="dim",   width=18)
    table.add_column("Accuracy /5",   justify="center")
    table.add_column("Brevity /5",    justify="center")
    table.add_column("Helpfulness /5", justify="center")
    table.add_column("Total /5",      justify="center", style="bold")
    table.add_column("N",             justify="center", style="dim")

    best_total = max(s.avg_total for s in summaries)

    for s in summaries:
        label_text = f"Prompt {s.label}"
        if s.avg_total == best_total:
            label_text += " 🏆"
        table.add_row(
            label_text,
            s.strategy,
            _colour_score(s.avg_accuracy),
            _colour_score(s.avg_brevity),
            _colour_score(s.avg_helpfulness),
            _colour_score(s.avg_total),
            str(s.n_evaluations),
        )

    console.print(table)


def print_detail_table(scores: list[EvaluationScore]) -> None:
    """Print every individual evaluation row."""
    console.print("\n[bold underline]Detailed Evaluations[/bold underline]\n")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Prompt", width=9)
    table.add_column("Test Case", width=12)
    table.add_column("Acc", justify="center", width=5)
    table.add_column("Brev", justify="center", width=5)
    table.add_column("Help", justify="center", width=5)
    table.add_column("Total", justify="center", width=6)
    table.add_column("Reasoning", style="dim")

    for s in sorted(scores, key=lambda x: (x.prompt_label, x.test_case_id)):
        snip = (s.reasoning[:90] + "…") if len(s.reasoning) > 90 else s.reasoning
        table.add_row(
            f"Prompt {s.prompt_label}",
            s.test_case_id,
            _colour_score(s.accuracy),
            _colour_score(s.brevity),
            _colour_score(s.helpfulness),
            _colour_score(s.total),
            snip,
        )

    console.print(table)


def print_winner_analysis(summaries: list[PromptSummary]) -> None:
    """Print a brief textual analysis of which prompt won and why."""
    console.rule()
    ranked = sorted(summaries, key=lambda s: s.avg_total, reverse=True)
    winner = ranked[0]
    console.print(
        f"\n[bold green]Winner: Prompt {winner.label} ({winner.strategy})[/bold green] "
        f"with avg total {winner.avg_total}/5\n"
    )

    # Explicit comparator functions — no dynamic getattr or string interpolation.
    best_accuracy    = max(summaries, key=lambda s: s.avg_accuracy)
    best_brevity     = max(summaries, key=lambda s: s.avg_brevity)
    best_helpfulness = max(summaries, key=lambda s: s.avg_helpfulness)

    for dim, best, score in [
        ("Accuracy",    best_accuracy,    best_accuracy.avg_accuracy),
        ("Brevity",     best_brevity,     best_brevity.avg_brevity),
        ("Helpfulness", best_helpfulness, best_helpfulness.avg_helpfulness),
    ]:
        console.print(
            f"  Best {dim}: Prompt {best.label} ({best.strategy}) — {score}/5"
        )
    console.print()


# ─── JSON Report ──────────────────────────────────────────────────────────────

def save_json_report(
    task: str,
    prompts: GeneratedPrompts,
    results: list[PromptResult],
    scores: list[EvaluationScore],
    summaries: list[PromptSummary],
    output_path: str,
) -> None:
    """Write the full evaluation report as a JSON file.

    The output path is resolved via pathlib to prevent directory traversal.
    """
    out = Path(output_path).resolve()

    # Only allow writing within the current working directory or a subdirectory.
    cwd = Path.cwd().resolve()
    try:
        out.relative_to(cwd)
    except ValueError:
        raise PermissionError(
            f"Output path '{output_path}' is outside the working directory. "
            "Refusing to write there for security reasons."
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "prompts": {
            "A": {"strategy": "Zero-shot",        "text": prompts.prompt_a},
            "B": {"strategy": "Chain-of-thought",  "text": prompts.prompt_b},
            "C": {"strategy": "Few-shot",          "text": prompts.prompt_c},
            "rationale": prompts.rationale,
        },
        "summary": [
            {
                "label":           s.label,
                "strategy":        s.strategy,
                "avg_accuracy":    s.avg_accuracy,
                "avg_brevity":     s.avg_brevity,
                "avg_helpfulness": s.avg_helpfulness,
                "avg_total":       s.avg_total,
                "n_evaluations":   s.n_evaluations,
            }
            for s in summaries
        ],
        "evaluations": [
            {
                "prompt_label":  s.prompt_label,
                "test_case_id":  s.test_case_id,
                "accuracy":      s.accuracy,
                "brevity":       s.brevity,
                "helpfulness":   s.helpfulness,
                "total":         s.total,
                "reasoning":     s.reasoning,
            }
            for s in sorted(scores, key=lambda x: (x.prompt_label, x.test_case_id))
        ],
        "raw_outputs": [
            {
                "prompt_label":    r.prompt_label,
                "test_case_id":    r.test_case_id,
                "test_case_input": r.test_case_input,
                "raw_output":      r.raw_output,
            }
            for r in sorted(results, key=lambda x: (x.prompt_label, x.test_case_id))
        ],
    }

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]✓ JSON report saved:[/green] {out}")
