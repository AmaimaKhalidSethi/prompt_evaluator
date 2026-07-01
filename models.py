"""
models.py — All Pydantic v2 data schemas for the Prompt Engineering Evaluator.

Security note:
  - Constraints (ge/le, max_length) are enforced at the model level.
  - These models are NEVER passed to LangChain's load()/loads() — only to
    with_structured_output(), which uses tool-calling, not deserialization.
  - This avoids CVE-2025-68664 (LangGrinch) and CVE-2026-44843 entirely.

Written for: pydantic>=2.0, langchain-core==1.4.8
"""

from typing import Literal

from pydantic import BaseModel, Field


# ─── Generator Output ─────────────────────────────────────────────────────────

class GeneratedPrompts(BaseModel):
    """Three strategically distinct prompts produced by the Generator Agent.

    The class name and field descriptions are injected into the tool-calling
    schema, so they guide the LLM. Keep them precise.
    """

    prompt_a: str = Field(
        description=(
            "Zero-shot prompt: a direct, concise instruction with no examples "
            "and no step-by-step guidance. Just tell the model what to do."
        ),
        min_length=10,
        max_length=1500,
    )
    prompt_b: str = Field(
        description=(
            "Chain-of-thought prompt: instructs the model to reason step by step "
            "before giving a final answer. Must contain explicit step-by-step guidance."
        ),
        min_length=10,
        max_length=1500,
    )
    prompt_c: str = Field(
        description=(
            "Few-shot prompt: includes exactly 2 concrete input→output examples "
            "that demonstrate the task format, followed by the actual instruction."
        ),
        min_length=10,
        max_length=2000,
    )
    rationale: str = Field(
        description=(
            "One sentence per prompt explaining the strategic difference between A, B, and C."
        ),
        max_length=600,
    )


# ─── Runner Output ────────────────────────────────────────────────────────────

class PromptResult(BaseModel):
    """Raw output from executing one (prompt, test_case) pair."""

    prompt_label: Literal["A", "B", "C"]   # enforced at construction
    prompt_text: str
    test_case_id: str
    test_case_input: str
    raw_output: str


# ─── Evaluator Output ─────────────────────────────────────────────────────────

class JudgeOutput(BaseModel):
    """Structured output returned directly by the Judge LLM.

    Labels (prompt_label, test_case_id) are NOT included here — the LLM
    only scores; labels are attached in Python to prevent label hallucination.
    """

    accuracy: int = Field(
        ge=1,
        le=5,
        description=(
            "1–5: How accurately the response addresses the task. "
            "1=completely wrong, 3=mostly correct, 5=perfectly accurate."
        ),
    )
    brevity: int = Field(
        ge=1,
        le=5,
        description=(
            "1–5: How concise the response is. "
            "1=extremely verbose/padded, 3=appropriate length, 5=perfectly tight."
        ),
    )
    helpfulness: int = Field(
        ge=1,
        le=5,
        description=(
            "1–5: How useful the response is for the task. "
            "1=unhelpful/misleading, 3=addresses core need, 5=maximally helpful."
        ),
    )
    reasoning: str = Field(
        description=(
            "One concise sentence explaining ALL three scores together."
        ),
        max_length=400,
    )


class EvaluationScore(BaseModel):
    """Full evaluation score with labels attached (assembled in Python)."""

    prompt_label: Literal["A", "B", "C"]
    test_case_id: str
    accuracy: int = Field(ge=1, le=5)      # mirrors JudgeOutput constraints
    brevity: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    reasoning: str

    @property
    def total(self) -> float:
        return round((self.accuracy + self.brevity + self.helpfulness) / 3, 2)


# ─── Report ───────────────────────────────────────────────────────────────────

class PromptSummary(BaseModel):
    """Aggregated stats for one prompt label across all test cases."""

    label: str
    strategy: str
    avg_accuracy: float
    avg_brevity: float
    avg_helpfulness: float
    avg_total: float
    n_evaluations: int
