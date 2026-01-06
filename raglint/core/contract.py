"""
Core contracts for raglint.

Minimal data structures for RAG output verification:
- Source: a retrieved document/chunk
- RagSample: answer + sources to verify
- CheckResult: pass/fail with evidence
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field


@dataclass
class Source:
    """
    A retrieved source document or chunk.
    
    Example:
        Source(
            id="doc_42",
            text="Revenue grew 12% in Q3 2024 to $450M.",
            metadata={"page": 3, "file": "quarterly_report.pdf"}
        )
    """
    
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Source(id={self.id!r}, text={preview!r})"


@dataclass
class RagSample:
    """
    A RAG output to verify: answer + retrieved sources.
    
    This is the primary input to all checks.
    
    Example:
        sample = RagSample(
            answer="Revenue grew 12% in Q3.",
            sources=[
                Source(id="doc_1", text="Q3 2024: Revenue reached $450M, up 12% from Q2.")
            ]
        )
    """
    
    answer: str
    sources: list[Source] = field(default_factory=list)
    
    @property
    def source_texts(self) -> list[str]:
        """All source texts concatenated for easy searching."""
        return [s.text for s in self.sources]
    
    @property
    def combined_sources(self) -> str:
        """All source texts as one string (for substring checks)."""
        return " ".join(self.source_texts)
    
    def __repr__(self) -> str:
        preview = self.answer[:50] + "..." if len(self.answer) > 50 else self.answer
        return f"RagSample(answer={preview!r}, sources={len(self.sources)})"


@dataclass
class CheckResult:
    """
    Result of running a single check.
    
    This is the return type for all checks.
    Use `passed` in assertions: `assert result.passed`
    Use `reasons` for CI output and debugging.
    
    Example:
        CheckResult(
            passed=False,
            score=0.3,
            reasons=["Entity 'Acme Corp' not found in sources"],
            evidence={"missing_entities": ["Acme Corp"], "matched_entities": ["Q3", "2024"]}
        )
    """
    
    passed: bool
    score: float | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    check_name: str = ""  # Filled by Check classes
    
    def __bool__(self) -> bool:
        """Allow `if result:` syntax."""
        return self.passed
    
    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        if self.reasons:
            return f"CheckResult({status}: {self.reasons[0]})"
        return f"CheckResult({status})"
    
    def explain(self) -> str:
        """Human-readable explanation for CI output."""
        lines = ["✓ PASS" if self.passed else "✗ FAIL"]
        
        if self.check_name:
            lines[0] = f"{lines[0]} [{self.check_name}]"
        
        if self.score is not None:
            lines.append(f"  Score: {self.score:.2f}")
        
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dict for JSON export / CI integration.
        
        Example:
            result.to_dict()
            # {"passed": False, "score": 0.3, "check": "grounded", ...}
        """
        return {
            "passed": self.passed,
            "score": self.score,
            "check": self.check_name or "unknown",
            "reasons": self.reasons,
            "evidence": self.evidence,
        }
    
    def to_json(self, indent: int | None = None) -> str:
        """
        Serialize to JSON string.
        
        Example:
            print(result.to_json(indent=2))
        """
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @property
    def exit_code(self) -> int:
        """
        Exit code for CLI/CI: 0 = pass, 1 = fail.
        
        Example:
            sys.exit(result.exit_code)
        """
        return 0 if self.passed else 1
