"""
agents/generator.py — Prompt Generator Agent

Takes a task description → returns 3 strategically distinct prompts via
LangChain's with_structured_output() (tool-calling, NOT deserialization).

Security:
  - API key injected at call time, never module-level global.
  - Task string is user-supplied but bounded by validate_task() in main.py
    before reaching here.
  - with_structured_output() uses tool-calling under the hood — no
    load()/loads() involved, so CVE-2025-68664 and CVE-2026-44843 do not apply.

Written for: langchain==1.3.11, langchain-core==1.4.8, langchain-groq==1.1.2
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from models import GeneratedPrompts

_DEFAULT_GENERATOR_MODEL = "qwen/qwen3.6-27b"

GENERATOR_SYSTEM = """\
You are a world-class prompt engineer. Given a task description, produce three
prompts that are STRATEGICALLY DIFFERENT — not just reworded versions of the same idea.

STRICT REQUIREMENTS:
  Prompt A — ZERO-SHOT
    • A single direct instruction sentence or two.
    • No examples. No "think step by step". No preamble.
    • Example pattern: "Summarise the following text in one sentence: {input}"

  Prompt B — CHAIN-OF-THOUGHT
    • Explicitly tell the model to reason through the problem before answering.
    • Must include phrasing like "Think step by step" or "First identify X, then Y".
    • Structure: reasoning instruction → then the task.

  Prompt C — FEW-SHOT
    • Include exactly 2 concrete input→output examples before the real task.
    • Format each example clearly: "Example 1\\nInput: ...\\nOutput: ..."
    • End with: "Now do the same for:\\nInput: {input}\\nOutput:"

Use {input} as the placeholder where the test case input will be inserted.
Make the strategy difference unmistakable — a reader should instantly see
which approach each prompt uses.\
"""


def _build_llm(api_key: str) -> object:
    """Construct a ChatGroq chain with structured output for GeneratedPrompts.

    Model name is read here (not at import time) so that load_dotenv() in
    main() has already populated os.environ before this is called.
    """
    model = os.environ.get("GENERATOR_MODEL", _DEFAULT_GENERATOR_MODEL)
    llm = ChatGroq(
        model=model,
        temperature=0.7,   # Some creativity for prompt diversity
        max_tokens=2000,
        timeout=30,        # Prevent indefinite hang on slow Groq responses
        groq_api_key=api_key,
        max_retries=2,
    )
    # with_structured_output uses tool-calling; no serialization CVEs apply.
    return llm.with_structured_output(GeneratedPrompts)


async def generate_prompts(task: str, api_key: str) -> GeneratedPrompts:
    """Generate 3 strategically distinct prompts for the given task.

    Args:
        task:    Validated task description from CLI input.
        api_key: Groq API key (gsk_...), validated before calling.

    Returns:
        GeneratedPrompts with prompt_a (zero-shot), prompt_b (CoT),
        prompt_c (few-shot), and rationale.

    Raises:
        ValueError: if the LLM returns None (e.g. refusal).
        Exception:  propagated from the LangChain / Groq API call.
    """
    chain = _build_llm(api_key)

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM),
        HumanMessage(content=f"Task description: {task}"),
    ]

    result: GeneratedPrompts | None = await chain.ainvoke(messages)

    if result is None:
        raise ValueError(
            "Generator returned no output. "
            "The model may have refused the request or hit a rate limit."
        )

    return result
