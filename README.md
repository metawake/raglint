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

## What raglint Catches vs. What It Doesn't

raglint focuses on **obvious failures** — the 80% of bad outputs that are clearly wrong:

| raglint catches | raglint doesn't catch |
|-----------------|----------------------|
| ✅ Hallucinated entities ("John Smith" not in sources) | ❌ Semantic correctness (entity is correct but wrong context) |
| ✅ Invented numbers ("15%" when sources say "12%") | ❌ Subtle hallucinations (paraphrased facts that sound right) |
| ✅ Template patterns ("TechCorp", "according to experts") | ❌ Source quality (are the sources themselves correct?) |
| ✅ Broken retrieval (answer unrelated to sources) | ❌ Answer completeness (did it fully answer the question?) |
| ✅ Not grounded (generic LLM output) | ❌ Factual truth (is "12%" the right number for this context?) |

### Why This Still Matters

Even though raglint doesn't catch everything, it provides **quality bounds**:

- **Lower bound:** If raglint fails, the output is definitely broken (no false negatives on obvious failures)
- **Upper bound:** If raglint passes, the output is "not obviously broken" (may still have subtle issues)

This is valuable because:
1. **Catches the most dangerous failures** — hallucinated entities and numbers are the #1 and #2 causes of RAG failures in production
2. **Zero cost regression detection** — catch quality drops before they reach users
3. **Complements deeper evaluation** — use raglint for fast checks, LLM-as-judge for critical paths
4. **Deterministic CI gates** — no flaky tests, no LLM costs, runs in milliseconds

Think of it like unit tests: they don't prove correctness, but they catch obvious bugs before integration testing.

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

## How It Works: Write Once, Run Repeatedly

raglint follows a simple workflow:

1. **Write your test suite once** — Customize assertions for your RAG pipeline
2. **Run repeatedly** — Same tests catch quality drops automatically

### Step 1: Write Your Test Suite

Create pytest tests with assertions tailored to your use case:

```python
# tests/test_rag_quality.py
import pytest
from raglint import grounded, numeric_grounded, no_placeholders

def test_financial_rag_quality():
    """Test that financial RAG outputs are grounded."""
    answer, sources = financial_rag("What was Q3 revenue?")
    
    assert grounded(answer, sources).passed
    assert numeric_grounded(answer, sources).passed
    assert no_placeholders(answer).passed

def test_entity_questions():
    """Test that entity questions return grounded entities."""
    answer, sources = rag.ask("Who is the CEO?")
    assert grounded(answer, sources).passed
```

### Step 2: Run Repeatedly (Regression Testing)

Run the same tests:
- **In CI** on every commit
- **When prompts change** to catch regressions
- **When models update** to verify quality
- **Manually** when debugging issues

```bash
# Run your test suite
pytest tests/test_rag_quality.py -v

# Same tests, different inputs each time
# Catches quality drops automatically
```

Same assertions, different inputs — that's regression testing.

## Working with Different RAG Output Formats

raglint works with common LLM output formats. It automatically extracts text from:

### JSON Format

```python
# LLM returns JSON
response = {
    "answer": "Revenue grew 12% to $450M.",
    "sources": [
        {"id": "doc_1", "text": "Q3 2024: Revenue reached $450M, up 12%."},
        {"id": "doc_2", "text": "Quarterly report shows strong growth."}
    ]
}

# raglint extracts text automatically
result = grounded(response["answer"], response["sources"])
```

### Dict Format (with metadata)

```python
# Sources as dicts
sources = [
    {"id": "doc_1", "text": "Q3: $450M revenue", "metadata": {"page": 3}},
    {"text": "Revenue grew 12%"}  # id optional
]

result = grounded(answer, sources)  # Works with dicts
```

### String Format

```python
# Simple string sources
sources = [
    "Q3 2024: Revenue reached $450M, up 12%.",
    "Quarterly report shows strong growth."
]

result = grounded(answer, sources)  # Works with strings
```

### Source Objects

```python
from raglint import Source

# Explicit Source objects
sources = [
    Source(id="doc_1", text="Q3: $450M revenue", metadata={"page": 3}),
    Source(id="doc_2", text="Revenue grew 12%")
]

result = grounded(answer, sources)
```

### Common LLM Q/A Formats

raglint extracts text from common patterns:

```python
# Pattern 1: Separate answer and sources
answer = "Revenue grew 12% to $450M."
sources = ["Q3 2024: Revenue reached $450M, up 12%."]

# Pattern 2: Structured response
response = {
    "content": "Revenue grew 12% to $450M.",
    "citations": [{"text": "Q3 2024: Revenue reached $450M, up 12%."}]
}
answer = response["content"]
sources = [c["text"] for c in response["citations"]]

# Pattern 3: With prefixes/markers
answer = "Answer: Revenue grew 12% to $450M."
sources = ["Source 1: Q3 2024: Revenue reached $450M, up 12%."]
# raglint normalizes text, so prefixes don't matter

result = grounded(answer, sources)  # Works with all formats
```

**Key point:** raglint normalizes text (lowercase, whitespace) before matching, so it works with various formatting conventions.

## API Reference

### Functional API

```python
grounded(answer, sources, threshold=0.8, fuzzy=False, fuzzy_threshold=0.85) -> CheckResult
cites_sources(answer, sources, threshold=0.1, ngram_size=3) -> CheckResult
no_placeholders(answer, patterns=None) -> CheckResult
numeric_grounded(answer, sources, threshold=0.8) -> CheckResult
```

**New in v0.3:** `grounded()` now supports `fuzzy=True` for matching entity variations.

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
GroundedCheck(threshold=0.8, fuzzy=False, fuzzy_threshold=0.85)
CitesSourcesCheck(threshold=0.1, ngram_size=3)
NoPlaceholdersCheck(patterns=None)
NumericGroundingCheck(threshold=0.8, require_any=True, tolerance=0.001)
```

**New in v0.3:** 
- `GroundedCheck` supports fuzzy matching via `fuzzy=True`
- `NumericGroundingCheck` normalizes numbers (M/B/K suffixes, percentages)

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

## v0.3.0: Improved Detection

### Better Entity Extraction

Now extracts more entity types:
- **Acronyms:** IBM, NASA, AWS, NVIDIA
- **camelCase brands:** openAI, iPhone, macOS, PostgreSQL
- **Hyphenated names:** Jean-Pierre, Hewlett-Packard, Rolls-Royce
- **Company suffixes:** Apple Inc, Google LLC, Microsoft Corporation
- **Contextual entities:** "works at Apple", "from Microsoft"

### Smart Number Normalization

Numbers now match across formats:
```python
# These all match now:
assert numbers_match("$1.5M", "1,500,000")  # ✓
assert numbers_match("15%", "0.15")          # ✓
assert numbers_match("2K", "2000")           # ✓
```

### Optional Fuzzy Matching (stdlib only)

Enable fuzzy matching for entity variations:
```python
# Without fuzzy: "Acme Corp" vs "Acme Corporation" = FAIL
# With fuzzy: matches at ~86% similarity = PASS
result = grounded(answer, sources, fuzzy=True, fuzzy_threshold=0.85)
```

Uses Python's `difflib.SequenceMatcher` — **zero external dependencies**.

## Philosophy

### What We Believe

1. **Most obvious RAG failures are catchable** — hallucinated entities, invented numbers, template patterns
2. **You don't need LLM to catch obvious problems** — deterministic checks are faster, cheaper, reproducible
3. **pytest is the right abstraction** — familiar, composable, CI-friendly
4. **Negative testing works** — "not obviously broken" is valuable even without proving "correct"

### What We Don't Do

- No LLM calls (deterministic only)
- No truth verification (that's a different problem)
- No golden answers (invariants only)
- No "interesting metrics" (pass/fail, not scores for dashboards)

## Extending raglint

### Want More Sophisticated NER?

raglint is intentionally dependency-free, using pattern-based extraction. If you need more sophisticated entity recognition, you can integrate spaCy or other NLP libraries:

```python
from raglint import Check, CheckResult, RagSample

class SpacyGroundedCheck(Check):
    """Custom grounded check using spaCy NER."""
    
    name = "spacy_grounded"
    
    def __init__(self, threshold: float = 0.8):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.threshold = threshold
    
    def run(self, sample: RagSample) -> CheckResult:
        # Extract entities with spaCy
        doc = self.nlp(sample.answer)
        entities = [ent.text for ent in doc.ents]
        
        if not entities:
            return CheckResult(passed=True, reasons=["No entities"], check_name=self.name)
        
        combined = sample.combined_sources.lower()
        matched = [e for e in entities if e.lower() in combined]
        missing = [e for e in entities if e.lower() not in combined]
        
        coverage = len(matched) / len(entities)
        passed = coverage >= self.threshold
        
        return CheckResult(
            passed=passed,
            score=coverage,
            reasons=[f"spaCy grounding: {coverage:.0%}"],
            evidence={"matched": matched, "missing": missing},
            check_name=self.name,
        )

# Usage
check = SpacyGroundedCheck()
result = check.run(sample)
```

**Why we don't include this by default:**
- Adds 400MB+ dependency (spaCy + model)
- Non-deterministic across spaCy versions (model updates change behavior)
- Overkill for obvious cases — "John Smith" doesn't need BERT

**When you might want it:**
- Domain-specific entities (medical, legal)
- Non-English languages
- Higher recall requirements
- Already using spaCy in your pipeline

**Contributions welcome!** If you build useful extensions, consider opening a PR or sharing them as a separate package.

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
