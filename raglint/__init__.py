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
)

__version__ = "0.2.0"

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
]
