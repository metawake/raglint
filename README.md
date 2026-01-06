# raglint

**Static analysis for RAG outputs. Catch hallucinations before production.**

raglint verifies *properties* of RAG answers — not truth, not correctness, just invariants that catch obviously bad outputs.

```python
from raglint import grounded, no_placeholders, numeric_grounded

def test_answer_is_grounded():
    answer = "Revenue grew 12% in Q3 to $450M."
    sources = ["Q3 2024: Revenue reached $450M, up 12% from Q2."]
    
    assert grounded(answer, sources).passed
    assert numeric_grounded(answer, sources).passed
    assert no_placeholders(answer).passed
```

## What It Does

| Check | What It Catches |
|-------|-----------------|
| `grounded()` | Hallucinated entities (names, companies not in sources) |
| `cites_sources()` | Generic LLM output / broken retrieval |
| `no_placeholders()` | Template patterns ("John Doe", "TechCorp", "recent study") |
| `numeric_grounded()` | Invented statistics (numbers not in sources) |

## What It Does NOT Do

- ❌ Verify truth or correctness
- ❌ Use LLM calls
- ❌ Require golden answers
- ❌ Catch subtle semantic errors

This is **negative testing**: "answer is not obviously broken."

## Where raglint Shines

| Scenario | Why raglint helps |
|----------|---------------------|
| **CI/CD regression** | Prompt changed? Model updated? Catch quality drops automatically, at zero cost |
| **Broken retrieval** | Sources empty or garbage → LLM hallucinates → raglint catches it |
| **Cost-optimized pipelines** | GPT-3.5, Mistral, Llama — safety net without extra LLM calls |
| **Scale** | 1% error rate × 10K requests/day = 100 bad answers. Automate the catch. |

```python
# In CI: catch when retrieval breaks or prompts degrade
def test_rag_pipeline_quality():
    answer, sources = rag_pipeline("What was Q3 revenue?")
    
    assert grounded(answer, sources).passed, "Entities not from sources!"
    assert len(sources) > 0, "Retrieval returned nothing!"
```

**Scope:** raglint is a fast, deterministic quality gate — like ESLint for RAG outputs. It catches obvious issues in milliseconds at zero cost. For deeper semantic analysis, use complementary tools.

## Installation

```bash
pip install raglint
```

Or from source:

```bash
git clone https://github.com/metawake/raglint.git
cd raglint
pip install -e .
```

## Quick Start

### Functional API (recommended for tests)

```python
from raglint import grounded, cites_sources, no_placeholders, numeric_grounded

# Your RAG pipeline
answer = rag.answer("What was Q3 revenue?")
sources = rag.retrieve("What was Q3 revenue?")

# Smoke tests
def test_grounding():
    result = grounded(answer, sources)
    assert result.passed, result.reasons

def test_no_placeholders():
    result = no_placeholders(answer)
    assert result.passed, result.reasons

def test_numbers_are_grounded():
    result = numeric_grounded(answer, sources)
    assert result.passed, result.reasons
```

### Object API (for configuration)

```python
from raglint import GroundedCheck, NumericGroundingCheck, RagSample, Source

# Configure thresholds
check = GroundedCheck(threshold=0.9)

sample = RagSample(
    answer="Acme Corp revenue grew 12%.",
    sources=[Source(id="doc_1", text="Q3: Revenue grew 12% to $450M.")]
)

result = check.run(sample)
print(result.explain())
# ✗ FAIL
#   Score: 0.00
#   - Entity grounding: 0% < 90%
#   - Missing: Acme Corp
```

### CheckSuite (combine checks)

```python
from raglint import CheckSuite, GroundedCheck, NumericGroundingCheck, NoPlaceholdersCheck

suite = CheckSuite([
    GroundedCheck(threshold=0.8),
    NumericGroundingCheck(),
    NoPlaceholdersCheck(),
])

result = suite.run(answer, sources)
assert result.passed, result.reasons
```

## API Reference

### Functional API

```python
grounded(answer, sources, threshold=0.8) -> CheckResult
cites_sources(answer, sources, threshold=0.1, ngram_size=3) -> CheckResult
no_placeholders(answer, patterns=None) -> CheckResult
numeric_grounded(answer, sources, threshold=0.8) -> CheckResult
```

### Data Classes

```python
@dataclass
class Source:
    id: str
    text: str
    metadata: dict = {}

@dataclass
class RagSample:
    answer: str
    sources: list[Source]

@dataclass
class CheckResult:
    passed: bool
    score: float | None
    reasons: list[str]
    evidence: dict
    check_name: str
    
    # Methods
    def to_dict() -> dict      # Structured output for logging/dashboards
    def to_json() -> str       # JSON string for CI artifacts  
    def explain() -> str       # Human-readable for logs
    exit_code: int             # 0 = pass, 1 = fail (for CLI/scripts)
```

### Check Classes

```python
GroundedCheck(threshold=0.8)
CitesSourcesCheck(threshold=0.1, ngram_size=3)
NoPlaceholdersCheck(patterns=None)
NumericGroundingCheck(threshold=0.8, require_any=True)
```

## Checks in Detail

### GroundedCheck

Extracts named entities from the answer (pattern-based, no ML) and verifies they appear in sources.

```python
answer = "John Smith at Acme Corp reported 12% growth."
sources = ["Q3 report shows 12% revenue growth."]

result = grounded(answer, sources)
# FAIL: "John Smith", "Acme Corp" not in sources
```

**Evidence:**
- `matched`: entities found in sources
- `missing`: entities not in sources
- `threshold`: configured threshold

### CitesSourcesCheck

Calculates n-gram overlap (Jaccard) between answer and sources.

```python
answer = "The weather is nice today."
sources = ["Q3 revenue reached $450M."]

result = cites_sources(answer, sources)
# FAIL: no overlap between answer and sources
```

**Evidence:**
- `overlap_score`: Jaccard similarity
- `overlapping_phrases`: matched n-grams

### NoPlaceholdersCheck

Detects placeholder patterns that indicate template output.

**Default patterns:**
- Generic names: "John Smith", "Jane Doe"
- Placeholder companies: "TechCorp", "DataSolutions"
- Vague citations: "recent study", "according to experts"
- Test data: "lorem", "foo", "example.com"

```python
answer = "According to experts, TechCorp is growing."
result = no_placeholders(answer)
# FAIL: "TechCorp", "according to experts" matched
```

### NumericGroundingCheck

Extracts numbers from the answer and verifies they appear in sources.

```python
answer = "Revenue grew 15% to $500M."
sources = ["Q3 revenue: $450M, up 12%."]

result = numeric_grounded(answer, sources)
# FAIL: 15%, $500M not in sources
```

**Evidence:**
- `matched`: numbers found in sources
- `missing`: numbers not in sources

## pytest Integration

```python
# tests/test_rag_quality.py

import pytest
from raglint import grounded, no_placeholders, numeric_grounded

@pytest.fixture
def rag_response():
    """Your RAG pipeline here."""
    return {
        "answer": "Revenue grew 12% in Q3.",
        "sources": ["Q3 2024: 12% revenue growth."]
    }

def test_entities_are_grounded(rag_response):
    result = grounded(rag_response["answer"], rag_response["sources"])
    assert result.passed, f"Ungrounded entities: {result.evidence.get('missing')}"

def test_no_placeholder_patterns(rag_response):
    result = no_placeholders(rag_response["answer"])
    assert result.passed, f"Placeholders found: {result.evidence.get('findings')}"

def test_numbers_are_sourced(rag_response):
    result = numeric_grounded(rag_response["answer"], rag_response["sources"])
    assert result.passed, f"Ungrounded numbers: {result.evidence.get('missing')}"
```

Run with pytest:

```bash
pytest tests/test_rag_quality.py -v
```

## CI Integration

### Structured Output

Every `CheckResult` provides machine-readable output:

```python
result = grounded(answer, sources)

# JSON for CI artifacts
print(result.to_json(indent=2))
# {
#   "passed": false,
#   "score": 0.0,
#   "check": "grounded",
#   "reasons": ["Entity grounding: 0% < 80%", "Missing: Acme Corp"],
#   "evidence": {"matched": [], "missing": ["Acme Corp"]}
# }

# Exit code for scripts
import sys
sys.exit(result.exit_code)  # 0 = pass, 1 = fail

# Dict for logging/dashboards
log_to_datadog(result.to_dict())
```

### GitHub Actions

```yaml
# .github/workflows/raglint.yml
name: RAG Quality Checks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install raglint pytest
      - run: pytest tests/test_rag_quality.py -v
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         RagSample                               │
│  ┌─────────────────┐    ┌─────────────────────────────────┐    │
│  │     answer      │    │            sources              │    │
│  │  "Revenue grew  │    │  [Source(text="Q3: $450M...")]  │    │
│  │   12% to $450M" │    │                                 │    │
│  └─────────────────┘    └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Check                                  │
│                                                                 │
│   GroundedCheck ──────► Extract entities → Match in sources     │
│   NumericGroundingCheck ► Extract numbers → Match in sources    │
│   NoPlaceholdersCheck ──► Pattern match → Detect templates      │
│   CitesSourcesCheck ────► N-gram overlap → Measure similarity   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CheckResult                               │
│                                                                 │
│   passed: bool ─────────► For assertions: assert result.passed  │
│   score: float ─────────► For thresholds: 0.0 to 1.0            │
│   reasons: list[str] ───► For CI output: human-readable         │
│   evidence: dict ───────► For debugging: matched/missing items  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Design principles:**

1. **No ML/NLP dependencies** — Pattern-based extraction keeps it fast and dependency-free
2. **Deterministic** — Same input always produces same output (reproducible in CI)
3. **Composable** — Checks are independent; combine with `CheckSuite` or run individually
4. **Evidence-first** — Every result explains *why* it passed/failed

## Philosophy

### What We Believe

1. **80% of bad RAG outputs are obviously bad** — hallucinated entities, invented numbers, template patterns
2. **You don't need LLM to catch obvious problems** — deterministic checks are faster, cheaper, reproducible
3. **pytest is the right abstraction** — familiar, composable, CI-friendly
4. **Negative testing works** — "not obviously broken" is valuable even without proving "correct"

### What We Don't Do

- No LLM calls (deterministic only)
- No truth verification (that's a different problem)
- No golden answers (invariants only)
- No "interesting metrics" (pass/fail, not scores for dashboards)

## Contributing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific tests
pytest tests/test_checks.py -v
```

## License

MIT

---

*raglint catches the obvious failures so you can focus on the hard problems.*
