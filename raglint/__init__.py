"""
rag-check — pytest-native quality gate for RAG systems.

Verifies RAG output properties without LLM calls:
- grounded(): entities in answer appear in sources
- cites_sources(): answer overlaps with sources
- no_placeholders(): no template patterns ("John Doe", "TechCorp")
- numeric_grounded(): numbers in answer appear in sources

Usage:
    from raglint import grounded, no_placeholders

    def test_answer_is_grounded():
        answer = rag_pipeline("What was Q3 revenue?")
        sources = retrieve("What was Q3 revenue?")
        
        assert grounded(answer, sources).passed
        assert no_placeholders(answer).passed
"""

from raglint.core.contract import Source, RagSample, CheckResult
from raglint.core.checks import (
    # Base
    Check,
    CheckSuite,
    # Checks
    GroundedCheck,
    CitesSourcesCheck,
    NoPlaceholdersCheck,
    NumericGroundingCheck,
    # Functional API
    grounded,
    cites_sources,
    no_placeholders,
    numeric_grounded,
    # Utilities
    extract_entities,
    extract_numbers,
    normalize_number,
    numbers_match,
    fuzzy_match,
    fuzzy_contains,
)

__version__ = "0.3.0"  # Bumped: improved entity extraction, number normalization, fuzzy matching

__all__ = [
    # Core contracts
    "Source",
    "RagSample",
    "CheckResult",
    # Base classes
    "Check",
    "CheckSuite",
    # Check classes
    "GroundedCheck",
    "CitesSourcesCheck",
    "NoPlaceholdersCheck",
    "NumericGroundingCheck",
    # Functional API (primary)
    "grounded",
    "cites_sources",
    "no_placeholders",
    "numeric_grounded",
    # Utilities
    "extract_entities",
    "extract_numbers",
    "normalize_number",
    "numbers_match",
    "fuzzy_match",
    "fuzzy_contains",
]
