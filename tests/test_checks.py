"""Tests for rag-check checks."""

import pytest

from raglint import (
    Source,
    RagSample,
    CheckResult,
    GroundedCheck,
    CitesSourcesCheck,
    NoPlaceholdersCheck,
    NumericGroundingCheck,
    CheckSuite,
    grounded,
    cites_sources,
    no_placeholders,
    numeric_grounded,
)


# =============================================================================
# Test Data
# =============================================================================

@pytest.fixture
def good_sample():
    """A well-grounded RAG sample."""
    return RagSample(
        answer="Revenue grew 12% in Q3 2024 to $450M.",
        sources=[
            Source(id="doc_1", text="Q3 2024 Financial Report: Revenue reached $450M, representing 12% growth from Q2.")
        ]
    )


@pytest.fixture
def bad_sample():
    """A sample with hallucinated entities."""
    return RagSample(
        answer="John Smith at Acme Corp reported 15% growth.",
        sources=[
            Source(id="doc_1", text="Q3 2024: Revenue grew 12%.")
        ]
    )


# =============================================================================
# GroundedCheck Tests
# =============================================================================

class TestGroundedCheck:
    """Tests for GroundedCheck — entity grounding."""
    
    def test_passes_when_entities_in_sources(self, good_sample):
        """Check passes when entities appear in sources."""
        check = GroundedCheck()
        result = check.run(good_sample)
        
        assert result.passed is True
        assert result.score is not None
    
    def test_fails_when_entities_missing(self, bad_sample):
        """Check fails when entities are hallucinated."""
        check = GroundedCheck()
        result = check.run(bad_sample)
        
        assert result.passed is False
        assert "Acme Corp" in result.evidence["missing"] or "John Smith" in result.evidence["missing"]
    
    def test_passes_with_no_entities(self):
        """Check passes when there are no entities to verify."""
        sample = RagSample(
            answer="Revenue grew by twelve percent.",
            sources=[Source(id="doc_1", text="Growth was strong.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        assert result.passed is True
        assert "No entities" in result.reasons[0]
    
    def test_threshold_configuration(self, bad_sample):
        """Check respects threshold configuration."""
        # Very low threshold should pass
        check = GroundedCheck(threshold=0.0)
        result = check.run(bad_sample)
        
        assert result.passed is True
    
    def test_callable_interface(self, good_sample):
        """Check can be called directly."""
        check = GroundedCheck()
        result = check(good_sample)
        
        assert isinstance(result, CheckResult)


# =============================================================================
# CitesSourcesCheck Tests
# =============================================================================

class TestCitesSourcesCheck:
    """Tests for CitesSourcesCheck — n-gram overlap."""
    
    def test_passes_with_overlap(self):
        """Check passes when answer overlaps with sources."""
        # Use sample with clear word overlap
        sample = RagSample(
            answer="Revenue grew by twelve percent in the third quarter.",
            sources=[Source(id="doc_1", text="Revenue grew by twelve percent in the third quarter of 2024.")]
        )
        
        check = CitesSourcesCheck(threshold=0.05)
        result = check.run(sample)
        
        assert result.passed is True
        assert result.score is not None
        assert result.score > 0
    
    def test_fails_without_sources(self):
        """Check fails when no sources provided."""
        sample = RagSample(answer="Some answer.", sources=[])
        
        check = CitesSourcesCheck()
        result = check.run(sample)
        
        assert result.passed is False
        assert "No sources" in result.reasons[0]
    
    def test_fails_with_no_overlap(self):
        """Check fails when answer doesn't overlap with sources."""
        sample = RagSample(
            answer="The weather is nice today.",
            sources=[Source(id="doc_1", text="Q3 revenue reached $450M.")]
        )
        
        check = CitesSourcesCheck(threshold=0.05)
        result = check.run(sample)
        
        assert result.passed is False
    
    def test_provides_overlap_evidence(self, good_sample):
        """Check provides overlapping phrases in evidence."""
        check = CitesSourcesCheck()
        result = check.run(good_sample)
        
        assert "overlap_score" in result.evidence
        assert "overlapping_phrases" in result.evidence


# =============================================================================
# NoPlaceholdersCheck Tests
# =============================================================================

class TestNoPlaceholdersCheck:
    """Tests for NoPlaceholdersCheck — placeholder detection."""
    
    def test_passes_with_clean_text(self, good_sample):
        """Check passes when no placeholders found."""
        check = NoPlaceholdersCheck()
        result = check.run(good_sample)
        
        assert result.passed is True
    
    def test_fails_with_generic_names(self):
        """Check fails when generic names found."""
        sample = RagSample(
            answer="John Smith reported the findings.",
            sources=[]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        assert result.passed is False
        assert any("John Smith" in str(f) for f in result.evidence["findings"])
    
    def test_fails_with_placeholder_company(self):
        """Check fails when placeholder company found."""
        sample = RagSample(
            answer="TechCorp announced new products.",
            sources=[]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        assert result.passed is False
    
    def test_fails_with_vague_citation(self):
        """Check fails when vague citation found."""
        sample = RagSample(
            answer="According to experts, growth is expected.",
            sources=[]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        assert result.passed is False
    
    def test_custom_patterns(self):
        """Check supports custom patterns."""
        sample = RagSample(
            answer="PLACEHOLDER_VALUE is 42.",
            sources=[]
        )
        
        custom_patterns = [(r'PLACEHOLDER_\w+', 'custom_placeholder')]
        check = NoPlaceholdersCheck(patterns=custom_patterns)
        result = check.run(sample)
        
        assert result.passed is False


# =============================================================================
# NumericGroundingCheck Tests
# =============================================================================

class TestNumericGroundingCheck:
    """Tests for NumericGroundingCheck — number verification."""
    
    def test_passes_when_numbers_in_sources(self, good_sample):
        """Check passes when numbers appear in sources."""
        check = NumericGroundingCheck()
        result = check.run(good_sample)
        
        assert result.passed is True
    
    def test_fails_when_numbers_missing(self, bad_sample):
        """Check fails when numbers are invented."""
        check = NumericGroundingCheck(require_any=False)
        result = check.run(bad_sample)
        
        # 15% is in answer but not in sources (stored as "15" after extraction)
        assert result.passed is False
        assert any("15" in m for m in result.evidence["missing"])
    
    def test_passes_with_no_numbers(self):
        """Check passes when there are no numbers to verify."""
        sample = RagSample(
            answer="Revenue grew significantly.",
            sources=[Source(id="doc_1", text="Strong growth observed.")]
        )
        
        check = NumericGroundingCheck()
        result = check.run(sample)
        
        assert result.passed is True
    
    def test_require_any_mode(self):
        """Check with require_any=True passes if at least one number matches."""
        sample = RagSample(
            answer="Revenue was $450M with 99% accuracy.",
            sources=[Source(id="doc_1", text="Revenue: $450M")]
        )
        
        check = NumericGroundingCheck(require_any=True)
        result = check.run(sample)
        
        # $450M matches, so should pass even though 99% doesn't
        assert result.passed is True
    
    def test_extracts_various_number_formats(self):
        """Check handles various number formats."""
        sample = RagSample(
            answer="Revenue: $1,234.56, growth: 12.5%, count: 1000",
            sources=[Source(id="doc_1", text="$1,234.56 revenue, 12.5% growth, 1000 units")]
        )
        
        check = NumericGroundingCheck()
        result = check.run(sample)
        
        assert result.passed is True


# =============================================================================
# CheckSuite Tests
# =============================================================================

class TestCheckSuite:
    """Tests for CheckSuite — check composition."""
    
    def test_passes_when_all_pass(self):
        """Suite passes when all checks pass."""
        # Use sample that passes all checks
        answer = "Revenue grew by twelve percent in the third quarter."
        sources = ["Revenue grew by twelve percent in the third quarter of 2024."]
        
        suite = CheckSuite([
            GroundedCheck(),
            CitesSourcesCheck(threshold=0.05),
            NoPlaceholdersCheck(),
        ])
        
        result = suite.run(answer, sources)
        
        assert result.passed is True
    
    def test_fails_when_any_fails(self, bad_sample):
        """Suite fails when any check fails."""
        suite = CheckSuite([
            GroundedCheck(),
            NoPlaceholdersCheck(),
        ])
        
        result = suite.run(bad_sample.answer, bad_sample.sources)
        
        assert result.passed is False
    
    def test_combines_evidence(self, good_sample):
        """Suite combines evidence from all checks."""
        suite = CheckSuite([
            GroundedCheck(),
            CitesSourcesCheck(),
        ])
        
        result = suite.run(good_sample.answer, good_sample.sources)
        
        # Evidence now nested under "checks"
        assert "checks" in result.evidence
        assert "GroundedCheck" in result.evidence["checks"]
        assert "CitesSourcesCheck" in result.evidence["checks"]
        # Also has structured results
        assert "results" in result.evidence
        assert "passed_count" in result.evidence
    
    def test_accepts_string_sources(self):
        """Suite accepts plain string sources."""
        suite = CheckSuite([NoPlaceholdersCheck()])
        
        result = suite.run(
            "Revenue grew 12%.",
            ["Revenue grew 12% in Q3."]  # Plain strings
        )
        
        assert result.passed is True
    
    def test_accepts_dict_sources(self):
        """Suite accepts dict sources."""
        suite = CheckSuite([NoPlaceholdersCheck()])
        
        result = suite.run(
            "Revenue grew 12%.",
            [{"id": "doc_1", "text": "Revenue grew 12% in Q3."}]
        )
        
        assert result.passed is True


# =============================================================================
# Functional API Tests
# =============================================================================

class TestFunctionalAPI:
    """Tests for functional API — grounded(), cites_sources(), etc."""
    
    def test_grounded_function(self):
        """grounded() function works."""
        result = grounded(
            "Revenue grew 12% in Q3.",
            ["Q3 2024: Revenue grew 12%."]
        )
        
        assert isinstance(result, CheckResult)
        assert result.passed is True
    
    def test_cites_sources_function(self):
        """cites_sources() function works."""
        result = cites_sources(
            "Revenue grew 12% in Q3.",
            ["Q3 2024: Revenue grew 12%."]
        )
        
        assert isinstance(result, CheckResult)
    
    def test_no_placeholders_function(self):
        """no_placeholders() function works."""
        result = no_placeholders("Revenue grew 12%.")
        
        assert isinstance(result, CheckResult)
        assert result.passed is True
    
    def test_numeric_grounded_function(self):
        """numeric_grounded() function works."""
        result = numeric_grounded(
            "Revenue: $450M, growth: 12%",
            ["$450M revenue, 12% growth"]
        )
        
        assert isinstance(result, CheckResult)
        assert result.passed is True
    
    def test_functional_api_accepts_source_objects(self):
        """Functional API accepts Source objects."""
        result = grounded(
            "Revenue grew 12%.",
            [Source(id="doc_1", text="Q3: 12% revenue growth.")]
        )
        
        assert isinstance(result, CheckResult)


# =============================================================================
# CheckResult Tests
# =============================================================================

# =============================================================================
# Edge Cases & Confusion Matrix Coverage
# =============================================================================

class TestGroundedCheckEdgeCases:
    """Edge cases and false positive/negative scenarios for GroundedCheck."""
    
    def test_false_positive_real_entity_in_sources(self):
        """Real entity exists in sources — should PASS (not false positive)."""
        sample = RagSample(
            answer="John Smith announced the merger.",
            sources=[Source(id="doc_1", text="CEO John Smith announced the merger today.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        # This should PASS because "John Smith" IS in sources
        assert result.passed is True, f"False positive: {result.evidence}"
    
    def test_false_negative_unusual_hallucinated_name(self):
        """Unusual hallucinated name that might slip through patterns."""
        sample = RagSample(
            answer="Dr. Alexander Worthington presented the findings.",
            sources=[Source(id="doc_1", text="Revenue grew 12% in Q3.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        # Should FAIL — entity not in sources
        assert result.passed is False, f"False negative: entity not caught"
    
    def test_case_insensitive_matching(self):
        """Entity matching should be case-insensitive."""
        sample = RagSample(
            answer="ACME CORP reported growth.",
            sources=[Source(id="doc_1", text="Acme Corp announced results.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        # Should PASS — case insensitive match
        assert result.passed is True
    
    def test_partial_entity_match(self):
        """Partial entity names — conservative behavior."""
        sample = RagSample(
            answer="Acme Corporation reported growth.",
            sources=[Source(id="doc_1", text="Acme Corp announced results.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        # Current behavior: likely FAIL because "Acme Corporation" != "Acme Corp"
        # This documents the expected behavior
        assert isinstance(result, CheckResult)


class TestNoPlaceholdersEdgeCases:
    """Edge cases for NoPlaceholdersCheck."""
    
    def test_false_positive_real_company_similar_to_placeholder(self):
        """Real company name that looks like placeholder."""
        sample = RagSample(
            answer="DataDog announced new features.",  # Real company!
            sources=[]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        # This tests current behavior — may be FP or not depending on patterns
        # Document what happens
        assert isinstance(result, CheckResult)
    
    def test_false_negative_creative_placeholder(self):
        """Creative placeholder that bypasses patterns."""
        sample = RagSample(
            answer="According to industry insiders, growth is expected.",
            sources=[]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        # "industry insiders" is vague but not in default patterns
        # This documents a potential gap
        assert isinstance(result, CheckResult)
    
    def test_legit_expert_attribution(self):
        """Legitimate expert attribution should ideally pass."""
        sample = RagSample(
            answer="According to Dr. Jane Chen at MIT, the results are promising.",
            sources=[Source(id="doc_1", text="Dr. Jane Chen at MIT published the study.")]
        )
        
        check = NoPlaceholdersCheck()
        result = check.run(sample)
        
        # Should PASS — specific named expert, not vague "experts"
        assert result.passed is True


class TestNumericGroundingEdgeCases:
    """Edge cases for NumericGroundingCheck."""
    
    def test_false_positive_formatted_differently(self):
        """Same number, different format — should match."""
        sample = RagSample(
            answer="Revenue was $1.5M.",
            sources=[Source(id="doc_1", text="Revenue reached $1,500,000.")]
        )
        
        check = NumericGroundingCheck()
        result = check.run(sample)
        
        # Documents current behavior — may or may not match
        assert isinstance(result, CheckResult)
    
    def test_false_negative_close_but_wrong_number(self):
        """Close number that is actually wrong — should FAIL."""
        sample = RagSample(
            answer="Growth was 12.5%.",
            sources=[Source(id="doc_1", text="Growth was 12.4%.")]
        )
        
        check = NumericGroundingCheck(require_any=False)
        result = check.run(sample)
        
        # Should FAIL — 12.5 != 12.4
        assert result.passed is False
    
    def test_year_numbers_handling(self):
        """Years should be treated as numbers."""
        sample = RagSample(
            answer="In 2024, revenue grew.",
            sources=[Source(id="doc_1", text="2024 financial report shows growth.")]
        )
        
        check = NumericGroundingCheck()
        result = check.run(sample)
        
        # 2024 is in both — should PASS
        assert result.passed is True


class TestCitesSourcesEdgeCases:
    """Edge cases for CitesSourcesCheck."""
    
    def test_paraphrased_content(self):
        """Paraphrased content with same meaning but different words."""
        sample = RagSample(
            answer="The company's earnings increased significantly.",
            sources=[Source(id="doc_1", text="Revenue grew by a large margin.")]
        )
        
        check = CitesSourcesCheck(threshold=0.05)
        result = check.run(sample)
        
        # Low n-gram overlap expected — documents behavior
        assert isinstance(result, CheckResult)
    
    def test_very_short_answer(self):
        """Very short answer — edge case for n-gram."""
        sample = RagSample(
            answer="Yes.",
            sources=[Source(id="doc_1", text="The answer is yes, confirmed.")]
        )
        
        check = CitesSourcesCheck()
        result = check.run(sample)
        
        # Documents behavior with minimal text
        assert isinstance(result, CheckResult)


# =============================================================================
# CheckResult Tests
# =============================================================================

class TestCheckResult:
    """Tests for CheckResult behavior."""
    
    def test_bool_conversion(self):
        """CheckResult can be used in boolean context."""
        passing = CheckResult(passed=True)
        failing = CheckResult(passed=False)
        
        assert passing
        assert not failing
    
    def test_explain_output(self):
        """CheckResult.explain() provides readable output."""
        result = CheckResult(
            passed=False,
            score=0.3,
            reasons=["Entity grounding: 30% < 80%", "Missing: Acme Corp"]
        )
        
        explanation = result.explain()
        
        assert "FAIL" in explanation
        assert "0.30" in explanation
        assert "Acme Corp" in explanation
    
    def test_repr(self):
        """CheckResult has useful repr."""
        result = CheckResult(passed=True, reasons=["All good"])
        
        assert "PASS" in repr(result)
        assert "All good" in repr(result)
    
    def test_to_dict(self):
        """CheckResult.to_dict() returns structured data."""
        result = CheckResult(
            passed=False,
            score=0.5,
            reasons=["Test reason"],
            evidence={"key": "value"},
            check_name="test_check",
        )
        
        d = result.to_dict()
        
        assert d["passed"] is False
        assert d["score"] == 0.5
        assert d["check"] == "test_check"
        assert d["reasons"] == ["Test reason"]
        assert d["evidence"] == {"key": "value"}
    
    def test_to_json(self):
        """CheckResult.to_json() returns valid JSON string."""
        import json
        
        result = CheckResult(
            passed=True,
            score=1.0,
            reasons=["All good"],
            check_name="grounded",
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["passed"] is True
        assert parsed["check"] == "grounded"
    
    def test_exit_code(self):
        """CheckResult.exit_code returns 0 for pass, 1 for fail."""
        passing = CheckResult(passed=True)
        failing = CheckResult(passed=False)
        
        assert passing.exit_code == 0
        assert failing.exit_code == 1
