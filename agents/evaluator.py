"""
agents/evaluator.py — Judge Agent

Scores one PromptResult on accuracy, brevity, and helpfulness (1–5 each)
using a calibrated rubric in the system prompt.

Design decisions:
  - temperature=0 for strict, reproducible scoring across evaluations.
  - Labels (prompt_label, test_case_id) are NOT inside JudgeOutput — the LLM
    only scores, and we attach labels in Python. This prevents label
    hallucination and keeps the schema minimal.
  - Uses with_structured_output() (tool-calling), not LangChain's load().

Security:
  - LLM output is treated as untrusted: Pydantic's ge/le constraints on
    JudgeOutput reject out-of-range scores before they reach the rest of the code.
  - Never passes user input to load_prompt() (CVE-2026-34070).
  - Never uses secrets_from_env=True (CVE-2025-68664).

Written for: langchain==1.3.11, langchain-core==1.4.8, langchain-groq==1.1.2
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from models import EvaluationScore, JudgeOutput, PromptResult

_DEFAULT_EVALUATOR_MODEL = "openai/gpt-oss-20b"

# ─── Rubric System Prompt ─────────────────────────────────────────────────────
# The rubric is explicit and anchored with calibration examples to make scoring
# consistent across all evaluations (satisfies "automated, consistent, explainable").

EVALUATOR_SYSTEM = """\
You are a rigorous prompt evaluation judge. Score each LLM response with precision.
Apply the rubric EXACTLY the same way every time — be consistent, not generous.

══════════════════════════════════════════════
SCORING RUBRIC
══════════════════════════════════════════════

ACCURACY (1–5): Does the response correctly address the task?
  1 = Completely wrong, off-topic, or hallucinated facts
  2 = Partially correct but with a major error or significant gap
  3 = Mostly correct; minor errors or one notable omission
  4 = Correct with only trivial imperfections
  5 = Perfectly accurate — nothing wrong, nothing missing

BREVITY (1–5): Is the response appropriately concise for the task?
  1 = Extremely verbose — repetitive, padded, full of filler sentences
  2 = More wordy than needed — could cut 30%+ without losing meaning
  3 = Appropriate length — no major waste, covers what's needed
  4 = Tight and efficient — every sentence earns its place
  5 = Perfectly concise — not a word wasted, yet nothing missing

HELPFULNESS (1–5): Would this response genuinely help someone do the task?
  1 = Unhelpful or actively misleading
  2 = Marginally helpful — misses what actually matters to the user
  3 = Helpful for the core need
  4 = Very helpful — goes slightly beyond the minimum in a useful way
  5 = Maximally helpful — directly solves the need in the most useful way

══════════════════════════════════════════════
CALIBRATION EXAMPLES
══════════════════════════════════════════════
Task: "Summarise in one sentence."
  Response: "Scientists created a flu vaccine with 95% efficacy."
  → accuracy=5, brevity=5, helpfulness=5  (concise, correct, perfect)

  Response: "This article discusses a new vaccine. It was developed by scientists
  who conducted trials. The trials showed the vaccine to be effective against flu."
  → accuracy=4, brevity=1, helpfulness=2  (accurate but three sentences, padded)

  Response: "The vaccine prevents all diseases."
  → accuracy=1, brevity=5, helpfulness=1  (concise but factually wrong)

══════════════════════════════════════════════
OUTPUT RULES
══════════════════════════════════════════════
- Use the FULL 1–5 range. Do not cluster scores around 3.
- Your "reasoning" field must be ONE sentence that mentions all three scores.
- Do not repeat the task or prompt in reasoning — just explain the scores.\
"""


def _build_evaluator(api_key: str) -> object:
    """Build the judge chain with temperature=0 for reproducibility.

    Model name is read here (not at import time) so that load_dotenv() in
    main() has already populated os.environ before this is called.
    """
    model = os.environ.get("EVALUATOR_MODEL", _DEFAULT_EVALUATOR_MODEL)
    llm = ChatGroq(
        model=model,
        temperature=0,      # Critical: must be 0 for consistent scoring
        max_tokens=300,
        timeout=30,         # Prevent indefinite hang on slow Groq responses
        groq_api_key=api_key,
        max_retries=2,
    )
    return llm.with_structured_output(JudgeOutput)


async def evaluate_output(
    result: PromptResult,
    task: str,
    api_key: str,
) -> EvaluationScore:
    """Score one PromptResult and return a fully labelled EvaluationScore.

    Args:
        result:  The raw output from one (prompt, test_case) run.
        task:    The original task description (for context).
        api_key: Groq API key.

    Returns:
        EvaluationScore with prompt_label, test_case_id, and all three scores.

    Raises:
        ValueError: if the judge returns None.
        Exception:  propagated from LangChain / Groq.
    """
    chain = _build_evaluator(api_key)

    user_content = (
        f"TASK BEING EVALUATED:\n{task}\n\n"
        f"PROMPT STRATEGY USED:\n{result.prompt_text}\n\n"
        f"TEST CASE INPUT:\n{result.test_case_input}\n\n"
        f"MODEL OUTPUT TO SCORE:\n{result.raw_output}\n\n"
        "Score the output above on accuracy, brevity, and helpfulness per the rubric."
    )

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM),
        HumanMessage(content=user_content),
    ]

    judge: JudgeOutput | None = await chain.ainvoke(messages)

    if judge is None:
        raise ValueError(
            f"Evaluator returned no output for "
            f"prompt={result.prompt_label}, test_case={result.test_case_id}."
        )

    # Attach labels in Python — not trusted from LLM output
    return EvaluationScore(
        prompt_label=result.prompt_label,
        test_case_id=result.test_case_id,
        accuracy=judge.accuracy,
        brevity=judge.brevity,
        helpfulness=judge.helpfulness,
        reasoning=judge.reasoning,
    )
