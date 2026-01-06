"""
Checks — pytest-native invariant checks for RAG outputs.

All checks:
- Are deterministic (no LLM, no randomness)
- Work with (answer, sources)
- Return CheckResult with pass/fail + evidence

Built-in checks:
- GroundedCheck: entities in answer must appear in sources
- CitesSourcesCheck: answer must overlap with sources (n-gram)
- NoPlaceholdersCheck: no placeholder patterns ("John Doe", "TechCorp")
- NumericGroundingCheck: numbers in answer must appear in sources
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

from raglint.core.contract import RagSample, CheckResult, Source


# =============================================================================
# Base Check Interface
# =============================================================================

class Check(ABC):
    """
    Base class for all checks.
    
    A check examines a RagSample and returns a CheckResult.
    
    To create a custom check:
    
        class MyCheck(Check):
            name = "my_check"
            
            def run(self, sample: RagSample) -> CheckResult:
                passed = ... # your logic
                return CheckResult(passed=passed, reasons=["..."], check_name=self.name)
    """
    
    name: str = "check"  # Override in subclasses
    
    @abstractmethod
    def run(self, sample: RagSample) -> CheckResult:
        """Run the check and return a result."""
        ...
    
    def __call__(self, sample: RagSample) -> CheckResult:
        """Allow check(sample) syntax."""
        return self.run(sample)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Entity Extraction (Simple, no ML dependencies)
# =============================================================================

# Patterns for extracting potential entities
ENTITY_PATTERNS = [
    # Capitalized multi-word names: "John Smith", "Acme Corporation"
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
    
    # Company suffixes: "Something Inc.", "Something Corp"
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Co|GmbH|AG|PLC)\.?)\b',
    
    # Patterns like "Dr. Name", "Prof. Name", "CEO Name"
    r'\b((?:Dr|Prof|Mr|Ms|Mrs|CEO|CTO|CFO)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
    
    # Year + study/report pattern: "2023 McKinsey Report"
    r'\b(\d{4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Report|Study|Survey|Analysis))\b',
]

# Common words to ignore (not entities)
IGNORE_WORDS = {
    'The', 'This', 'That', 'These', 'Those', 'There', 'Here',
    'What', 'When', 'Where', 'Which', 'Who', 'Why', 'How',
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'However', 'Therefore', 'Furthermore', 'Additionally', 'Moreover',
    'According', 'Based', 'Given', 'Despite', 'Although',
    'First', 'Second', 'Third', 'Finally', 'Next', 'Then',
    'Revenue', 'Sales', 'Profit', 'Growth', 'Market', 'Company',
}


def extract_entities(text: str) -> list[str]:
    """
    Extract potential named entities from text.
    
    Uses pattern matching — no ML/NER dependencies.
    Designed for high recall (over-extracts) for verification.
    """
    entities = []
    seen = set()
    
    for pattern in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text):
            entity_text = match.group(1) if match.lastindex else match.group(0)
            entity_text = entity_text.strip()
            
            # Skip short or ignored
            if len(entity_text) < 3:
                continue
            if entity_text in IGNORE_WORDS:
                continue
            if entity_text.lower() in seen:
                continue
            
            seen.add(entity_text.lower())
            entities.append(entity_text)
    
    return entities


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


# =============================================================================
# GroundedCheck — entities in answer must appear in sources
# =============================================================================

class GroundedCheck(Check):
    """
    Check that named entities in the answer appear in sources.
    
    Catches hallucinated names, companies, citations that the LLM invented.
    
    Args:
        threshold: Minimum fraction of entities that must be grounded (default: 0.8)
    
    Example:
        >>> check = GroundedCheck(threshold=0.8)
        >>> result = check.run(sample)
        >>> assert result.passed
    """
    
    name = "grounded"
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def run(self, sample: RagSample) -> CheckResult:
        entities = extract_entities(sample.answer)
        
        if not entities:
            return CheckResult(
                passed=True,
                score=1.0,
                reasons=["No entities to verify"],
                evidence={"entities": [], "matched": [], "missing": []},
                check_name=self.name,
            )
        
        combined = normalize_text(sample.combined_sources)
        
        matched = []
        missing = []
        
        for entity in entities:
            if normalize_text(entity) in combined:
                matched.append(entity)
            else:
                missing.append(entity)
        
        coverage = len(matched) / len(entities)
        passed = coverage >= self.threshold
        
        if passed:
            reasons = [f"Entity grounding: {coverage:.0%} ({len(matched)}/{len(entities)})"]
        else:
            reasons = [
                f"Entity grounding: {coverage:.0%} < {self.threshold:.0%}",
                f"Missing: {', '.join(missing[:3])}" + (f" (+{len(missing)-3} more)" if len(missing) > 3 else ""),
            ]
        
        return CheckResult(
            passed=passed,
            score=coverage,
            reasons=reasons,
            evidence={"matched": matched, "missing": missing, "threshold": self.threshold},
            check_name=self.name,
        )


# =============================================================================
# CitesSourcesCheck — answer must overlap with sources
# =============================================================================

def compute_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


class CitesSourcesCheck(Check):
    """
    Check that the answer shares content with sources (n-gram overlap).
    
    Catches generic LLM outputs or broken retrieval.
    
    Args:
        threshold: Minimum Jaccard overlap ratio (default: 0.1)
        ngram_size: N-gram size for overlap calculation (default: 3)
    
    Example:
        >>> check = CitesSourcesCheck(threshold=0.1)
        >>> result = check.run(sample)
        >>> assert result.passed
    """
    
    name = "cites_sources"
    
    def __init__(self, threshold: float = 0.1, ngram_size: int = 3):
        self.threshold = threshold
        self.ngram_size = ngram_size
    
    def run(self, sample: RagSample) -> CheckResult:
        if not sample.sources:
            return CheckResult(
                passed=False,
                score=0.0,
                reasons=["No sources provided"],
                evidence={"overlap_score": 0.0},
                check_name=self.name,
            )
        
        answer_ngrams = compute_ngrams(sample.answer, self.ngram_size)
        source_ngrams = compute_ngrams(sample.combined_sources, self.ngram_size)
        
        if not answer_ngrams or not source_ngrams:
            return CheckResult(
                passed=False,
                score=0.0,
                reasons=["Insufficient text for n-gram analysis"],
                evidence={"overlap_score": 0.0},
                check_name=self.name,
            )
        
        intersection = answer_ngrams & source_ngrams
        union = answer_ngrams | source_ngrams
        
        jaccard = len(intersection) / len(union) if union else 0.0
        passed = jaccard >= self.threshold
        
        # Find top overlapping spans for evidence
        overlapping_phrases = [" ".join(ngram) for ngram in list(intersection)[:5]]
        
        if passed:
            reasons = [f"Source overlap: {jaccard:.1%}"]
        else:
            reasons = [f"Source overlap: {jaccard:.1%} < {self.threshold:.1%}"]
        
        return CheckResult(
            passed=passed,
            score=jaccard,
            reasons=reasons,
            evidence={
                "overlap_score": jaccard,
                "overlapping_phrases": overlapping_phrases,
                "threshold": self.threshold,
            },
            check_name=self.name,
        )


# =============================================================================
# NoPlaceholdersCheck — no placeholder patterns
# =============================================================================

# Default placeholder patterns
DEFAULT_PLACEHOLDERS = [
    # Generic person names
    (r'\b(?:John|Jane|Bob|Alice|Tom|Mary)\s+(?:Smith|Doe|Johnson|Williams|Brown)\b', "generic_name"),
    
    # Placeholder company names
    (r'\b(?:Tech|Data|Cloud|Cyber|Info|Net|Web|Digi)(?:Corp|Tech|Co|Solutions|Systems|Works)\b', "placeholder_company"),
    
    # Acme-style names
    (r'\bAcme\s+\w+\b', "acme_pattern"),
    
    # Vague citations
    (r'\b(?:a\s+)?(?:recent|new|latest)\s+(?:study|research|report|survey)\b', "vague_citation"),
    
    # Anonymous attribution
    (r'\baccording to\s+(?:experts|researchers|scientists|analysts|sources)\b', "anonymous_attribution"),
    
    # Lorem ipsum / test data
    (r'\b(?:lorem|ipsum|foo|bar|baz|test|example|sample)\b', "test_data"),
    
    # Placeholder URLs/emails
    (r'\bexample\.(?:com|org|net)\b', "placeholder_url"),
    (r'\b\w+@example\.(?:com|org)\b', "placeholder_email"),
]


class NoPlaceholdersCheck(Check):
    """
    Check that the answer contains no placeholder patterns.
    
    Catches template-like outputs from LLMs.
    
    Args:
        patterns: List of (regex, label) tuples. Defaults to common placeholders.
    
    Example:
        >>> check = NoPlaceholdersCheck()
        >>> result = check.run(sample)
        >>> assert result.passed  # Fails if "John Doe" or "TechCorp" found
    """
    
    name = "no_placeholders"
    
    def __init__(self, patterns: list[tuple[str, str]] | None = None):
        self.patterns = patterns or DEFAULT_PLACEHOLDERS
    
    def run(self, sample: RagSample) -> CheckResult:
        findings = []
        
        for pattern, label in self.patterns:
            matches = re.findall(pattern, sample.answer, re.IGNORECASE)
            for match in matches:
                findings.append({"text": match, "type": label})
        
        # Deduplicate
        seen = set()
        unique = []
        for f in findings:
            key = f["text"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        passed = len(unique) == 0
        
        if passed:
            reasons = ["No placeholder patterns detected"]
        else:
            examples = [f["text"] for f in unique[:3]]
            reasons = [f"Placeholder patterns found: {', '.join(examples)}"]
        
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasons=reasons,
            evidence={"findings": unique, "count": len(unique)},
            check_name=self.name,
        )


# =============================================================================
# NumericGroundingCheck — numbers in answer must appear in sources
# =============================================================================

# Pattern for extracting numbers (integers, decimals, percentages, currency)
NUMBER_PATTERN = r'''
    (?:
        \$?\d{1,3}(?:,\d{3})*(?:\.\d+)?[MBK]?  |  # Currency: $450M, $1,234.56
        \d+(?:\.\d+)?%                          |  # Percentages: 12%, 3.5%
        \d{1,3}(?:,\d{3})+                      |  # Large numbers: 1,000,000
        \d+(?:\.\d+)?                              # Plain numbers: 42, 3.14
    )
'''


def extract_numbers(text: str) -> list[str]:
    """Extract numeric values from text."""
    pattern = re.compile(NUMBER_PATTERN, re.VERBOSE)
    return pattern.findall(text)


def normalize_number(num_str: str) -> str:
    """Normalize number for comparison: remove formatting."""
    # Remove currency symbols and suffixes
    normalized = num_str.replace('$', '').replace(',', '')
    # Handle M/B/K suffixes (but keep for matching)
    return normalized.lower()


class NumericGroundingCheck(Check):
    """
    Check that numbers in the answer appear in sources.
    
    Catches the most dangerous hallucinations: invented statistics.
    
    Args:
        threshold: Minimum fraction of numbers that must be grounded (default: 0.8)
        require_any: If True, at least one number must match. If False, missing numbers = fail.
    
    Example:
        >>> check = NumericGroundingCheck()
        >>> result = check.run(sample)
        >>> assert result.passed  # Fails if "12%" is in answer but not in sources
    """
    
    name = "numeric_grounded"
    
    def __init__(self, threshold: float = 0.8, require_any: bool = True):
        self.threshold = threshold
        self.require_any = require_any
    
    def run(self, sample: RagSample) -> CheckResult:
        answer_numbers = extract_numbers(sample.answer)
        
        if not answer_numbers:
            return CheckResult(
                passed=True,
                score=1.0,
                reasons=["No numbers to verify"],
                evidence={"numbers": [], "matched": [], "missing": []},
                check_name=self.name,
            )
        
        source_numbers = extract_numbers(sample.combined_sources)
        source_normalized = {normalize_number(n) for n in source_numbers}
        
        matched = []
        missing = []
        
        for num in answer_numbers:
            if normalize_number(num) in source_normalized:
                matched.append(num)
            else:
                missing.append(num)
        
        # Handle require_any: if at least one matches, that's often OK
        if self.require_any and matched:
            coverage = 1.0
            passed = True
        else:
            coverage = len(matched) / len(answer_numbers) if answer_numbers else 1.0
            passed = coverage >= self.threshold
        
        if passed:
            reasons = [f"Numeric grounding: {len(matched)}/{len(answer_numbers)} numbers verified"]
        else:
            reasons = [
                f"Numeric grounding: {coverage:.0%} < {self.threshold:.0%}",
                f"Ungrounded numbers: {', '.join(missing[:3])}",
            ]
        
        return CheckResult(
            passed=passed,
            score=coverage,
            reasons=reasons,
            evidence={"matched": matched, "missing": missing, "threshold": self.threshold},
            check_name=self.name,
        )


# =============================================================================
# CheckSuite — compose multiple checks
# =============================================================================

class CheckSuite:
    """
    Compose multiple checks into a single runnable suite.
    
    Example:
        suite = CheckSuite([
            GroundedCheck(),
            NumericGroundingCheck(),
            NoPlaceholdersCheck(),
        ])
        
        result = suite.run(answer, sources)
        assert result.passed
    """
    
    def __init__(self, checks: list[Check]):
        self.checks = checks
    
    def run(self, answer: str, sources: list[Source] | list[dict] | list[str]) -> CheckResult:
        """
        Run all checks and return combined result.
        
        Args:
            answer: The RAG answer to verify
            sources: List of Source objects, dicts, or plain strings
        
        Returns:
            CheckResult with passed=True only if all checks pass
        """
        sample = self._make_sample(answer, sources)
        
        all_reasons = []
        all_evidence = {}
        results_list = []
        all_passed = True
        total_score = 0.0
        
        for check in self.checks:
            result = check.run(sample)
            all_reasons.extend(result.reasons)
            all_evidence[check.__class__.__name__] = result.evidence
            results_list.append(result.to_dict())
            
            if not result.passed:
                all_passed = False
            
            if result.score is not None:
                total_score += result.score
        
        avg_score = total_score / len(self.checks) if self.checks else 1.0
        
        return CheckResult(
            passed=all_passed,
            score=avg_score,
            reasons=all_reasons,
            evidence={
                "checks": all_evidence,
                "results": results_list,
                "passed_count": sum(1 for r in results_list if r["passed"]),
                "total_count": len(results_list),
            },
            check_name="suite",
        )
    
    def _make_sample(self, answer: str, sources: list) -> RagSample:
        """Convert various source formats to RagSample."""
        source_objects = []
        
        for i, src in enumerate(sources):
            if isinstance(src, Source):
                source_objects.append(src)
            elif isinstance(src, dict):
                source_objects.append(Source(
                    id=src.get("id", f"source_{i}"),
                    text=src.get("text", str(src)),
                    metadata=src.get("metadata", {}),
                ))
            else:
                source_objects.append(Source(id=f"source_{i}", text=str(src)))
        
        return RagSample(answer=answer, sources=source_objects)
    
    def __repr__(self) -> str:
        check_names = [c.__class__.__name__ for c in self.checks]
        return f"CheckSuite({check_names})"


# =============================================================================
# Functional API — thin wrappers for pytest
# =============================================================================

def grounded(answer: str, sources: list, threshold: float = 0.8) -> CheckResult:
    """
    Check that entities in the answer are grounded in sources.
    
    Example:
        assert grounded(answer, sources).passed
    """
    sample = _make_sample(answer, sources)
    return GroundedCheck(threshold=threshold).run(sample)


def cites_sources(answer: str, sources: list, threshold: float = 0.1, ngram_size: int = 3) -> CheckResult:
    """
    Check that the answer overlaps with sources (n-gram).
    
    Example:
        assert cites_sources(answer, sources).passed
    """
    sample = _make_sample(answer, sources)
    return CitesSourcesCheck(threshold=threshold, ngram_size=ngram_size).run(sample)


def no_placeholders(answer: str, patterns: list[tuple[str, str]] | None = None) -> CheckResult:
    """
    Check that the answer contains no placeholder patterns.
    
    Example:
        assert no_placeholders(answer).passed
    """
    sample = RagSample(answer=answer, sources=[])
    return NoPlaceholdersCheck(patterns=patterns).run(sample)


def numeric_grounded(answer: str, sources: list, threshold: float = 0.8) -> CheckResult:
    """
    Check that numbers in the answer appear in sources.
    
    Example:
        assert numeric_grounded(answer, sources).passed
    """
    sample = _make_sample(answer, sources)
    return NumericGroundingCheck(threshold=threshold).run(sample)


def _make_sample(answer: str, sources: list) -> RagSample:
    """Helper: convert sources to RagSample."""
    source_objects = []
    for i, src in enumerate(sources):
        if isinstance(src, Source):
            source_objects.append(src)
        elif isinstance(src, dict):
            source_objects.append(Source(
                id=src.get("id", f"source_{i}"),
                text=src.get("text", str(src)),
                metadata=src.get("metadata", {}),
            ))
        else:
            source_objects.append(Source(id=f"source_{i}", text=str(src)))
    return RagSample(answer=answer, sources=source_objects)
