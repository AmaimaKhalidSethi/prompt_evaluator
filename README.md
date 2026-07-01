# Prompt Engineering Evaluator

**Week 1, Project 1-I-B — CalderR Internship Program**

A CLI tool that takes a task description, automatically generates 3 strategically distinct prompts using an LLM, runs all 3 against test cases in parallel, and scores each output on Accuracy, Brevity, and Helpfulness using a calibrated judge agent.

---

## Architecture

```
CLI Input (--task, --cases)
    │
    ▼
[Input Validation]  ← length, pattern, path-traversal checks
    │
    ▼
[Generator Agent]   ← 1 LLM call, structured output (GeneratedPrompts)
  Produces:
    Prompt A — Zero-shot
    Prompt B — Chain-of-thought
    Prompt C — Few-shot
    │
    ▼
[Parallel Runner]   ← asyncio.gather(), 3 × N concurrent LLM calls
  Semaphore-bounded (MAX_CONCURRENT=6) to respect rate limits
    │
    ▼
[Parallel Evaluator] ← asyncio.gather(), 3 × N concurrent LLM calls
  temperature=0 for reproducibility
  Calibrated rubric in system prompt
    │
    ▼
[Reporter]          ← Rich console tables + JSON file
```

**Models used (June 2026):**
- Generator: `qwen/qwen3.6-27b` (replaces deprecated `qwen/qwen3-32b`)
- Runner + Evaluator: `openai/gpt-oss-20b` (replaces deprecated `llama-3.1-8b-instant`)

---

## Setup

**Prerequisites:** Python ≥ 3.10, a free [Groq API key](https://console.groq.com).

```bash
# Clone and enter the project
cd prompt_evaluator

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
```

---

## Usage

```bash
# Basic run — uses test_cases.json in the current directory
python main.py --task "Summarise a news article in one sentence"

# Custom test cases and output file
python main.py \
  --task "Explain quantum entanglement to a 10-year-old" \
  --cases my_cases.json \
  --output quantum_results.json

# Debug task
python main.py \
  --task "Find and fix the bug in this Python function" \
  --cases debug_cases.json
```

**CLI flags:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--task` | Yes | — | Task description (10–500 chars) |
| `--cases` | No | `test_cases.json` | Path to test cases JSON |
| `--output` | No | `results.json` | Path for JSON report |

---

## Test Cases Format

```json
[
  {
    "id": "tc1",
    "input": "The text or content the prompt will operate on."
  },
  {
    "id": "tc2",
    "input": "Another test input."
  }
]
```

Each `input` replaces the `{input}` placeholder in the generated prompts.

---

## Output

**Console:** Rich tables showing per-test-case scores and aggregate comparison.

**JSON report (`results.json`):** Full structured output including:
- Generated prompt texts for A, B, C
- Every individual evaluation (label, test_case_id, accuracy, brevity, helpfulness, reasoning)
- Aggregated averages per prompt strategy
- All raw LLM outputs

```json
{
  "generated_at": "2026-06-24T10:23:41Z",
  "task": "Summarise a news article in one sentence",
  "prompts": {
    "A": { "strategy": "Zero-shot", "text": "..." },
    "B": { "strategy": "Chain-of-thought", "text": "..." },
    "C": { "strategy": "Few-shot", "text": "..." }
  },
  "summary": [...],
  "evaluations": [...],
  "raw_outputs": [...]
}
```

---

## Skills Learned

| Skill | Where applied |
|-------|---------------|
| **Parallel LLM calls** | `runner.py` — `asyncio.gather()` with semaphore concurrency control |
| **Structured output with Pydantic** | `models.py` + `with_structured_output()` in both agents |
| **Evaluation design** | `agents/evaluator.py` — calibrated rubric, temperature=0, anchored 1–5 scale |
| **LangChain LCEL** | `ChatGroq(...).with_structured_output(Model)` — the standard LCEL chain pattern |
| **Security-first dev** | CVE research, input validation, path-traversal prevention, secret management |

---

---

# Findings Report

## Methodology

The evaluator was run on **5 different tasks**, each with 5 test cases, producing **75 total evaluations** (5 tasks × 3 prompts × 5 test cases). Scores are on a 1–5 scale. The judge runs at `temperature=0` to ensure consistency; the same rubric anchored with calibration examples is used for every evaluation.

**Three prompt strategies compared:**
- **Prompt A (Zero-shot):** Direct, concise instruction with no examples and no reasoning guidance.
- **Prompt B (Chain-of-thought):** Explicit "think step by step" instruction before answering.
- **Prompt C (Few-shot):** Two concrete input→output examples embedded before the actual task.

---

## Task 1: News Summarisation

**Task:** `"Summarise a news article in one sentence"`

| Prompt | Strategy | Accuracy | Brevity | Helpfulness | Total |
|--------|----------|----------|---------|-------------|-------|
| A | Zero-shot | 4.6 | 4.8 | 4.4 | **4.60** 🏆 |
| B | Chain-of-thought | 4.8 | 2.4 | 4.2 | 3.80 |
| C | Few-shot | 4.6 | 3.6 | 4.4 | 4.20 |

**Key finding:** Zero-shot dominated on a straightforward extraction task. Chain-of-thought hurt brevity badly — the model wrote multiple reasoning steps before reaching the summary, producing 4–5 sentences when the task asked for one. Few-shot struck a middle ground, but the examples in the prompt added overhead that the model sometimes mirrored in length.

---

## Task 2: Code Debugging

**Task:** `"Identify the bug in this Python function and explain the fix"`

| Prompt | Strategy | Accuracy | Brevity | Helpfulness | Total |
|--------|----------|----------|---------|-------------|-------|
| A | Zero-shot | 3.2 | 4.4 | 3.2 | 3.60 |
| B | Chain-of-thought | 4.8 | 3.0 | 4.8 | **4.20** 🏆 |
| C | Few-shot | 4.2 | 3.4 | 4.2 | 3.93 |

**Key finding:** Chain-of-thought reversed the winner. For debugging, reasoning through the code line by line led to substantially better accuracy and helpfulness. Zero-shot often gave the correct fix without adequate explanation, scoring well on brevity but poorly on helpfulness (users couldn't understand *why* the fix worked). Few-shot examples helped, but two examples could not cover the diversity of bug patterns in the test cases.

---

## Task 3: Conceptual Explanation

**Task:** `"Explain quantum entanglement to a 10-year-old"`

| Prompt | Strategy | Accuracy | Brevity | Helpfulness | Total |
|--------|----------|----------|---------|-------------|-------|
| A | Zero-shot | 4.0 | 3.8 | 3.8 | 3.87 |
| B | Chain-of-thought | 4.2 | 2.6 | 4.0 | 3.60 |
| C | Few-shot | 4.6 | 3.8 | 4.8 | **4.40** 🏆 |

**Key finding:** Few-shot won cleanly. The examples in Prompt C showed the model what an age-appropriate explanation looks like — using analogies and simple language — and the model replicated that style reliably. Zero-shot explanations were often technically accurate but too formal. Chain-of-thought produced over-structured responses with visible reasoning steps, which a child would find confusing.

---

## Task 4: Professional Email Drafting

**Task:** `"Write a professional email declining a meeting invitation politely"`

| Prompt | Strategy | Accuracy | Brevity | Helpfulness | Total |
|--------|----------|----------|---------|-------------|-------|
| A | Zero-shot | 4.2 | 4.6 | 4.4 | **4.40** 🏆 |
| B | Chain-of-thought | 4.2 | 2.8 | 3.8 | 3.60 |
| C | Few-shot | 4.4 | 3.4 | 4.6 | 4.13 |

**Key finding:** Zero-shot and few-shot were both strong. Emails are a familiar format that modern LLMs handle well without structural guidance, so the additional scaffolding of CoT added verbosity without accuracy gains. Chain-of-thought again produced long meta-commentary ("I will start with a polite opener, then…") before the actual email. Few-shot was close — the examples helped set the right tone — but zero-shot was surprisingly clean and concise.

---

## Task 5: Structured Data Conversion

**Task:** `"Convert this JSON object to a CSV row with these headers: name, age, city"`

| Prompt | Strategy | Accuracy | Brevity | Helpfulness | Total |
|--------|----------|----------|---------|-------------|-------|
| A | Zero-shot | 3.6 | 4.2 | 3.6 | 3.80 |
| B | Chain-of-thought | 4.4 | 3.0 | 4.4 | 3.93 |
| C | Few-shot | 4.8 | 4.0 | 4.8 | **4.53** 🏆 |

**Key finding:** Few-shot won decisively on a format-sensitive task. When the output must conform to an exact structure (CSV row, specific header order), showing the model a concrete example proved far more reliable than verbal instructions. Zero-shot frequently put values in the wrong column order or added unwanted quotes. Chain-of-thought improved accuracy over zero-shot (the step-by-step process helped the model map fields correctly) but Few-shot got there more efficiently.

---

## Cross-Task Summary

| Task | Winner | Why |
|------|--------|-----|
| News Summarisation | Zero-shot | Simple extraction; CoT adds wasteful reasoning steps |
| Code Debugging | Chain-of-thought | Reasoning through code line-by-line improves accuracy |
| Conceptual Explanation | Few-shot | Examples demonstrate appropriate style and tone |
| Email Drafting | Zero-shot | Familiar format; LLMs need no structural scaffolding |
| Data Conversion | Few-shot | Format-sensitive; showing the exact output structure removes ambiguity |

**Overall averages across all 5 tasks:**

| Strategy | Avg Accuracy | Avg Brevity | Avg Helpfulness | Avg Total |
|----------|-------------|-------------|-----------------|-----------|
| Zero-shot | 3.92 | 4.36 | 3.88 | 4.05 |
| Chain-of-thought | 4.48 | 2.76 | 4.24 | 3.83 |
| Few-shot | 4.52 | 3.64 | 4.56 | **4.24** |

---

## Key Conclusions

**1. No single strategy dominates across all tasks.**
Few-shot has the highest average total (4.24/5) but loses badly on simple extraction tasks where it adds unnecessary overhead. Choosing a prompt strategy should be task-specific.

**2. Chain-of-thought consistently hurts brevity (-1.6 points vs zero-shot).**
Every task showed CoT producing longer, more verbose output. This is the clearest and most consistent pattern in the data. CoT is worth the verbosity cost only when accuracy is the primary concern (debugging, reasoning tasks) and brevity can be traded off.

**3. Task type is the most predictive factor for prompt strategy:**
- **Extraction / summarisation tasks:** Use zero-shot. The task is unambiguous; extra scaffolding adds noise.
- **Reasoning / debugging tasks:** Use chain-of-thought. Step-by-step reasoning prevents logical errors.
- **Style-sensitive / format-sensitive tasks:** Use few-shot. Examples calibrate tone, structure, and output format more reliably than any verbal instruction.

**4. Evaluation is consistent and explainable.**
The judge uses temperature=0 and a rubric with calibration anchors. Rescoring the same outputs returned identical scores in manual spot-checks. Every score includes a reasoning field, making disagreements reviewable.

**5. Meaningful differences are visible in the data.**
The gap between best and worst strategy reaches 0.73 total points per task (Task 1: 4.60 vs 3.80). The differences are large enough to be actionable and are not noise.

---

## Security Notes

This project was built following the `security-first-dev` workflow. CVE research was conducted before any code was written.

**CVEs mitigated:**

| CVE | CVSS | Description | Mitigation |
|-----|------|-------------|------------|
| CVE-2025-68664 | 9.3 | LangGrinch — serialization injection via `dumps()`/`loads()` | Never use `load()/loads()` on LLM output; `secrets_from_env` defaults to False |
| CVE-2026-34070 | 7.5 | Path traversal in `load_prompt()` | Never call `load_prompt()` with user-controlled paths |
| CVE-2026-44843 | — | Unsafe deserialization via broad `allowed_objects` | Never pass `allowed_objects="all"` to `load()` |

**Additional security measures:**
- All user inputs validated (length, pattern, JSON schema) before any network call
- Output path resolved with `pathlib` and checked against CWD to prevent traversal
- API key read from `.env` only; never logged or printed
- Internal errors go to `evaluator.log` only; generic messages shown in console
- No `subprocess`, no `shell=True`, no `eval()` anywhere in the codebase
- `.env` in `.gitignore`
