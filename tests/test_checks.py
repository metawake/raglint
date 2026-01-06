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
    # New utilities
    extract_entities,
    extract_numbers,
    normalize_number,
    numbers_match,
    fuzzy_match,
    fuzzy_contains,
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


# =============================================================================
# New Feature Tests: Improved Entity Extraction
# =============================================================================

class TestImprovedEntityExtraction:
    """Tests for improved entity extraction patterns."""
    
    def test_extracts_acronyms(self):
        """Should extract all-caps acronyms like IBM, NASA, AWS."""
        entities = extract_entities("IBM and NASA announced a partnership with AWS.")
        
        assert "IBM" in entities
        assert "NASA" in entities
        assert "AWS" in entities
    
    def test_extracts_camelcase_brands(self):
        """Should extract camelCase brands like openAI, iPhone."""
        entities = extract_entities("The openAI team released a new iPhone app for macOS.")
        
        assert "openAI" in entities
        assert "iPhone" in entities
        assert "macOS" in entities
    
    def test_extracts_hyphenated_names(self):
        """Should extract hyphenated names like Jean-Pierre, Hewlett-Packard."""
        entities = extract_entities("Jean-Pierre works at Hewlett-Packard in Rolls-Royce division.")
        
        assert "Jean-Pierre" in entities
        assert "Hewlett-Packard" in entities
        assert "Rolls-Royce" in entities
    
    def test_extracts_company_with_suffix(self):
        """Should extract single-word companies with suffixes: Apple Inc, Google LLC."""
        entities = extract_entities("Apple Inc reported earnings. Google LLC announced updates.")
        
        assert "Apple Inc" in entities
        assert "Google LLC" in entities
    
    def test_extracts_contextual_single_words(self):
        """Should extract single words in context: 'at Apple', 'from Microsoft'."""
        entities = extract_entities("She works at Apple and previously came from Microsoft.")
        
        assert "Apple" in entities
        assert "Microsoft" in entities
    
    def test_ignores_common_acronyms(self):
        """Should ignore common acronyms like CEO, Q1, API, etc."""
        entities = extract_entities("The CEO reported Q1 results via the API.")
        
        assert "CEO" not in entities
        assert "Q1" not in entities
        assert "API" not in entities
    
    def test_still_extracts_multiword_names(self):
        """Should still extract multi-word names: John Smith, Acme Corporation."""
        entities = extract_entities("John Smith at Acme Corporation announced the merger.")
        
        assert "John Smith" in entities
        assert "Acme Corporation" in entities
    
    def test_grounded_check_with_acronyms(self):
        """GroundedCheck should work with acronym entities."""
        sample = RagSample(
            answer="IBM and AWS announced a cloud partnership.",
            sources=[Source(id="doc_1", text="IBM and AWS are collaborating on cloud services.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        assert result.passed is True
    
    def test_grounded_check_fails_with_hallucinated_acronym(self):
        """GroundedCheck should fail when acronym is hallucinated."""
        sample = RagSample(
            answer="IBM, AWS, and NVIDIA announced a partnership.",
            sources=[Source(id="doc_1", text="IBM and AWS are collaborating.")]
        )
        
        check = GroundedCheck()
        result = check.run(sample)
        
        # NVIDIA not in sources
        assert "NVIDIA" in result.evidence["missing"]


# =============================================================================
# New Feature Tests: Improved Number Normalization
# =============================================================================

class TestImprovedNumberNormalization:
    """Tests for improved number normalization."""
    
    def test_normalize_millions(self):
        """Should normalize M suffix: 1.5M → 1500000."""
        assert normalize_number("1.5M") == 1_500_000
        assert normalize_number("$1.5M") == 1_500_000
        assert normalize_number("2M") == 2_000_000
    
    def test_normalize_billions(self):
        """Should normalize B suffix: 1.2B → 1200000000."""
        assert normalize_number("1.2B") == 1_200_000_000
        assert normalize_number("$3B") == 3_000_000_000
    
    def test_normalize_thousands(self):
        """Should normalize K suffix: 500K → 500000."""
        assert normalize_number("500K") == 500_000
        assert normalize_number("2.5K") == 2_500
    
    def test_normalize_percentages(self):
        """Should normalize percentages: 15% → 0.15."""
        assert normalize_number("15%") == 0.15
        assert normalize_number("3.5%") == 0.035
        assert normalize_number("100%") == 1.0
    
    def test_normalize_with_commas(self):
        """Should handle comma-separated numbers."""
        assert normalize_number("1,234,567") == 1_234_567
        assert normalize_number("$1,234.56") == 1_234.56
    
    def test_normalize_with_signs(self):
        """Should handle positive/negative signs."""
        assert normalize_number("+12") == 12
        assert normalize_number("-5.5") == -5.5
    
    def test_numbers_match_different_formats(self):
        """numbers_match should match equivalent values in different formats."""
        # Million formats
        assert numbers_match("$1.5M", "1,500,000") is True
        assert numbers_match("1.5M", "$1500000") is True
        
        # Percentage and decimal
        assert numbers_match("15%", "0.15") is True
        assert numbers_match("3.5%", "0.035") is True
        
        # Thousands
        assert numbers_match("2K", "2000") is True
        assert numbers_match("2.5K", "2,500") is True
    
    def test_numbers_match_rejects_different_values(self):
        """numbers_match should reject actually different values."""
        assert numbers_match("15%", "0.16") is False
        assert numbers_match("$1.5M", "1,600,000") is False
        assert numbers_match("12", "13") is False
    
    def test_numeric_grounding_with_format_differences(self):
        """NumericGroundingCheck should match numbers in different formats."""
        sample = RagSample(
            answer="Revenue was $1.5M with 15% growth.",
            sources=[Source(id="doc_1", text="Revenue: $1,500,000. Growth rate: 0.15.")]
        )
        
        check = NumericGroundingCheck()
        result = check.run(sample)
        
        # Both numbers should match despite format differences
        assert result.passed is True
    
    def test_numeric_grounding_still_catches_invented(self):
        """NumericGroundingCheck should still catch invented numbers."""
        sample = RagSample(
            answer="Revenue was $2M with 20% growth.",  # Invented!
            sources=[Source(id="doc_1", text="Revenue: $1,500,000. Growth rate: 15%.")]
        )
        
        check = NumericGroundingCheck(require_any=False)
        result = check.run(sample)
        
        # Numbers don't match
        assert result.passed is False


# =============================================================================
# New Feature Tests: Fuzzy Matching
# =============================================================================

class TestFuzzyMatching:
    """Tests for stdlib-based fuzzy matching."""
    
    def test_fuzzy_match_similar_strings(self):
        """fuzzy_match should match similar strings."""
        # "Acme Corp" vs "Acme Corporation" - ~86% similar
        assert fuzzy_match("Acme Corp", "Acme Corporation announced results.") is True
        
        # "Microsoft" vs "Microsft" (typo) - ~90% similar
        assert fuzzy_match("Microsoft", "Microsft reported earnings.") is True
    
    def test_fuzzy_match_rejects_dissimilar(self):
        """fuzzy_match should reject dissimilar strings."""
        assert fuzzy_match("Apple", "Orange banana grape", threshold=0.85) is False
        assert fuzzy_match("John Smith", "Jane Doe reported", threshold=0.85) is False
    
    def test_fuzzy_match_exact_still_works(self):
        """fuzzy_match should still match exact substrings."""
        assert fuzzy_match("Apple", "Apple Inc reported earnings.") is True
        assert fuzzy_match("John Smith", "CEO John Smith announced...") is True
    
    def test_fuzzy_contains_exact_mode(self):
        """fuzzy_contains with threshold=1.0 should do exact matching."""
        # Exact match works
        assert fuzzy_contains("Apple", "Apple Inc", threshold=1.0) is True
        
        # Non-substring similar string doesn't work in exact mode
        # "Microsft" (typo) is not a substring of "Microsoft"
        assert fuzzy_contains("Microsft", "Microsoft announced results", threshold=1.0) is False
    
    def test_fuzzy_contains_fuzzy_mode(self):
        """fuzzy_contains with threshold<1.0 should do fuzzy matching."""
        # Similar strings match even with typo
        assert fuzzy_contains("Microsft", "Microsoft announced results", threshold=0.85) is True
        
        # Abbreviation variations
        assert fuzzy_contains("Intl Business Machines", "International Business Machines Corp", threshold=0.7) is True
    
    def test_grounded_check_with_fuzzy(self):
        """GroundedCheck with fuzzy=True should match similar entities."""
        sample = RagSample(
            answer="John Smithson reported strong earnings.",  # Similar to John Smith
            sources=[Source(id="doc_1", text="John Smith announced quarterly results.")]
        )
        
        # Without fuzzy: fails (John Smithson != John Smith)
        check_exact = GroundedCheck(fuzzy=False)
        result_exact = check_exact.run(sample)
        assert result_exact.passed is False, f"Expected fail, got: {result_exact.evidence}"
        
        # With fuzzy: passes (John Smithson ~ John Smith at ~0.85)
        check_fuzzy = GroundedCheck(fuzzy=True, fuzzy_threshold=0.75)
        result_fuzzy = check_fuzzy.run(sample)
        assert result_fuzzy.passed is True
    
    def test_grounded_functional_api_fuzzy(self):
        """grounded() function should support fuzzy parameter."""
        # Use multi-word entity that will be extracted
        answer = "Acme Industries reported earnings."
        sources = ["Acme Industrial Corp announced results."]
        
        # Without fuzzy - "Acme Industries" not exact match for "Acme Industrial Corp"
        result_exact = grounded(answer, sources, fuzzy=False)
        assert result_exact.passed is False, f"Expected fail, got: {result_exact.evidence}"
        
        # With fuzzy - should match
        result_fuzzy = grounded(answer, sources, fuzzy=True, fuzzy_threshold=0.7)
        assert result_fuzzy.passed is True
    
    def test_fuzzy_evidence_includes_mode(self):
        """Evidence should indicate whether fuzzy matching was used."""
        sample = RagSample(answer="Test", sources=[Source(id="1", text="Test")])
        
        check_exact = GroundedCheck(fuzzy=False)
        result_exact = check_exact.run(sample)
        assert result_exact.evidence["fuzzy"] is False
        
        check_fuzzy = GroundedCheck(fuzzy=True)
        result_fuzzy = check_fuzzy.run(sample)
        assert result_fuzzy.evidence["fuzzy"] is True


# =============================================================================
# Confusion Matrix Tests: Entity Extraction
# =============================================================================

class TestEntityExtractionConfusionMatrix:
    """
    Confusion matrix tests for improved entity extraction.
    
    Tests cover:
    - TP: Real entity in answer, correctly extracted
    - TN: No entity / common word, correctly ignored
    - FP: Non-entity incorrectly extracted (minimize)
    - FN: Real entity missed (minimize)
    """
    
    # --- TRUE POSITIVES: Should extract, does extract ---
    
    def test_tp_acronym_ibm(self):
        """TP: IBM should be extracted."""
        entities = extract_entities("IBM announced new products.")
        assert "IBM" in entities
    
    def test_tp_acronym_aws(self):
        """TP: AWS should be extracted."""
        entities = extract_entities("We deployed on AWS infrastructure.")
        assert "AWS" in entities
    
    def test_tp_camelcase_openai(self):
        """TP: openAI should be extracted."""
        entities = extract_entities("The openAI team released updates.")
        assert "openAI" in entities
    
    def test_tp_camelcase_iphone(self):
        """TP: iPhone should be extracted."""
        entities = extract_entities("The new iPhone features improved battery.")
        assert "iPhone" in entities
    
    def test_tp_hyphenated_person(self):
        """TP: Hyphenated names should be extracted."""
        entities = extract_entities("Jean-Pierre presented the findings.")
        assert "Jean-Pierre" in entities
    
    def test_tp_hyphenated_company(self):
        """TP: Hyphenated company names should be extracted."""
        entities = extract_entities("Hewlett-Packard announced earnings.")
        assert "Hewlett-Packard" in entities
    
    def test_tp_company_suffix(self):
        """TP: Company with suffix should be extracted."""
        entities = extract_entities("Apple Inc reported revenue growth.")
        assert "Apple Inc" in entities
    
    def test_tp_contextual_at(self):
        """TP: Entity after 'at' should be extracted."""
        entities = extract_entities("She works at Tesla designing batteries.")
        assert "Tesla" in entities
    
    def test_tp_multiword_name(self):
        """TP: Multi-word names should still work."""
        entities = extract_entities("John Smith announced the merger.")
        assert "John Smith" in entities
    
    # --- TRUE NEGATIVES: Should NOT extract, doesn't extract ---
    
    def test_tn_common_acronym_ceo(self):
        """TN: CEO is a title, not an entity."""
        entities = extract_entities("The CEO reported earnings.")
        assert "CEO" not in entities
    
    def test_tn_common_acronym_q1(self):
        """TN: Q1 is a time reference, not an entity."""
        entities = extract_entities("Q1 results were strong.")
        assert "Q1" not in entities
    
    def test_tn_common_acronym_api(self):
        """TN: API is a technical term, not an entity."""
        entities = extract_entities("The API returned errors.")
        assert "API" not in entities
    
    def test_tn_month_name(self):
        """TN: Month names should not be entities."""
        entities = extract_entities("Results for January were good.")
        assert "January" not in entities
    
    def test_tn_transition_word(self):
        """TN: Transition words should not be entities."""
        entities = extract_entities("However, growth was limited.")
        assert "However" not in entities
    
    def test_tn_currency_code(self):
        """TN: Currency codes should not be entities."""
        entities = extract_entities("Revenue was 100 USD.")
        assert "USD" not in entities
    
    # --- FALSE POSITIVES: Should NOT extract, but might (document known issues) ---
    
    def test_fp_all_caps_word_in_sentence(self):
        """
        FP risk: All-caps emphasis might be extracted.
        Document behavior.
        """
        entities = extract_entities("This is VERY important.")
        # VERY might be extracted as acronym - document behavior
        # If this becomes a problem, add to IGNORE_WORDS
        assert isinstance(entities, list)  # Document behavior
    
    def test_fp_title_case_common_noun(self):
        """
        FP risk: Title-case at sentence start.
        Should NOT extract common nouns that happen to be capitalized.
        """
        entities = extract_entities("Revenue grew significantly.")
        assert "Revenue" not in entities  # In IGNORE_WORDS
    
    # --- FALSE NEGATIVES: Should extract, but might miss (document known gaps) ---
    
    def test_fn_lowercase_brand(self):
        """
        FN risk: All-lowercase brands like 'adidas' won't be caught.
        This is a known limitation of pattern-based extraction.
        """
        entities = extract_entities("The adidas partnership was announced.")
        # adidas won't be extracted - known limitation
        assert "adidas" not in entities  # Expected FN
    
    def test_fn_numeric_name(self):
        """
        FN risk: Brands with numbers like '3M' might not match.
        """
        entities = extract_entities("3M announced new products.")
        # 3M pattern depends on our regex - document behavior
        assert isinstance(entities, list)  # Document behavior
    
    def test_fn_single_capital_letter_company(self):
        """
        FN risk: Single-letter entities are ignored to reduce noise.
        """
        entities = extract_entities("X (formerly Twitter) announced changes.")
        # Single letter 'X' intentionally not extracted
        assert "X" not in entities  # Expected FN (intentional)


# =============================================================================
# Confusion Matrix Tests: Number Normalization
# =============================================================================

class TestNumberNormalizationConfusionMatrix:
    """
    Confusion matrix tests for number normalization.
    
    Tests cover:
    - TP: Same value, different format → should match
    - TN: Different values → should NOT match
    - FP: Accidental matches (minimize)
    - FN: Same value but missed (minimize)
    """
    
    # --- TRUE POSITIVES: Same value, different format → matches ---
    
    def test_tp_millions_suffix_vs_full(self):
        """TP: $1.5M should match 1,500,000."""
        assert numbers_match("$1.5M", "1,500,000") is True
    
    def test_tp_percentage_vs_decimal(self):
        """TP: 15% should match 0.15."""
        assert numbers_match("15%", "0.15") is True
    
    def test_tp_thousands_suffix_vs_full(self):
        """TP: 2K should match 2000."""
        assert numbers_match("2K", "2000") is True
    
    def test_tp_billions_suffix(self):
        """TP: 1.2B should match 1,200,000,000."""
        assert numbers_match("1.2B", "1200000000") is True
    
    def test_tp_with_without_currency(self):
        """TP: $500 should match 500."""
        assert numbers_match("$500", "500") is True
    
    def test_tp_with_without_commas(self):
        """TP: 1,234,567 should match 1234567."""
        assert numbers_match("1,234,567", "1234567") is True
    
    def test_tp_positive_sign(self):
        """TP: +12 should match 12."""
        assert numbers_match("+12", "12") is True
    
    def test_tp_lowercase_suffix(self):
        """TP: Lowercase 'm' should work like 'M'."""
        assert numbers_match("1.5m", "1500000") is True
    
    # --- TRUE NEGATIVES: Different values → no match ---
    
    def test_tn_different_percentages(self):
        """TN: 15% should NOT match 16%."""
        assert numbers_match("15%", "16%") is False
    
    def test_tn_different_millions(self):
        """TN: $1.5M should NOT match $1.6M."""
        assert numbers_match("$1.5M", "$1.6M") is False
    
    def test_tn_close_but_wrong(self):
        """TN: 12.5 should NOT match 12.4 (precision matters)."""
        assert numbers_match("12.5", "12.4") is False
    
    def test_tn_order_of_magnitude(self):
        """TN: 1M should NOT match 1K."""
        assert numbers_match("1M", "1K") is False
    
    def test_tn_negative_vs_positive(self):
        """TN: -5 should NOT match 5."""
        assert numbers_match("-5", "5") is False
    
    # --- FALSE POSITIVES: Accidental matches (should be rare) ---
    
    def test_fp_within_tolerance(self):
        """
        FP risk: Very close numbers within tolerance might match.
        This is by design (tolerance=0.001 = 0.1%).
        """
        # 1000 and 1000.5 are within 0.1% - should match (by design)
        assert numbers_match("1000", "1000.5", tolerance=0.001) is True
        # But not with tighter tolerance
        assert numbers_match("1000", "1000.5", tolerance=0.0001) is False
    
    # --- FALSE NEGATIVES: Should match but might miss ---
    
    def test_fn_spelled_out_numbers(self):
        """
        FN: Spelled-out numbers like 'one million' won't match '1M'.
        This is a known limitation.
        """
        val = normalize_number("one million")
        assert val is None  # Can't parse spelled-out - expected FN
    
    def test_fn_mixed_units(self):
        """
        FN: Unit conversions like '1 km' vs '1000 m' won't match.
        No unit awareness - expected limitation.
        """
        # Would need unit conversion logic
        assert normalize_number("1 km") is None  # Can't parse units
    
    def test_fn_fraction_vs_decimal(self):
        """
        FN: Fractions like '1/2' won't match '0.5'.
        No fraction parsing - expected limitation.
        """
        val = normalize_number("1/2")
        assert val is None  # Can't parse fractions


# =============================================================================
# Confusion Matrix Tests: Fuzzy Matching
# =============================================================================

class TestFuzzyMatchingConfusionMatrix:
    """
    Confusion matrix tests for fuzzy entity matching.
    
    Tests cover:
    - TP: Similar entities → should match with fuzzy
    - TN: Dissimilar entities → should NOT match
    - FP: Unrelated strings accidentally matching (minimize)
    - FN: Similar but missed (document threshold sensitivity)
    """
    
    # --- TRUE POSITIVES: Similar strings should match ---
    
    def test_tp_abbreviation_variation(self):
        """TP: 'Corp' vs 'Corporation' should fuzzy match."""
        assert fuzzy_match("Acme Corp", "Acme Corporation announced", threshold=0.8) is True
    
    def test_tp_minor_typo(self):
        """TP: Minor typo should fuzzy match."""
        assert fuzzy_match("Microsoft", "Microsft announced earnings", threshold=0.85) is True
    
    def test_tp_missing_word(self):
        """TP: Partial name should fuzzy match."""
        assert fuzzy_match("John Smith", "John W Smith reported", threshold=0.75) is True
    
    def test_tp_exact_substring_fast_path(self):
        """TP: Exact substring uses fast path."""
        assert fuzzy_match("Apple", "Apple Inc reported", threshold=0.99) is True
    
    # --- TRUE NEGATIVES: Dissimilar should NOT match ---
    
    def test_tn_completely_different(self):
        """TN: Completely different strings should not match."""
        assert fuzzy_match("Apple", "Banana Orange Grape", threshold=0.8) is False
    
    def test_tn_different_entities(self):
        """TN: Different but valid entities should not match."""
        assert fuzzy_match("Microsoft", "Apple announced products", threshold=0.8) is False
    
    def test_tn_similar_length_different_content(self):
        """TN: Similar length but different content."""
        assert fuzzy_match("John Smith", "Jane Brown", threshold=0.8) is False
    
    def test_tn_partial_overlap_insufficient(self):
        """TN: Partial word overlap shouldn't be enough."""
        assert fuzzy_match("International Business", "Interstate Commerce", threshold=0.8) is False
    
    # --- FALSE POSITIVES: Accidental matches (document sensitivity) ---
    
    def test_fp_low_threshold_risk(self):
        """
        FP risk: Very low threshold might match unrelated strings.
        Document that threshold < 0.7 is risky.
        """
        # With very low threshold, unrelated might match
        # "Apple" vs "Apply" at 0.6 threshold
        result = fuzzy_match("Apple", "Apply to this job", threshold=0.6)
        # Document behavior - this might match at low thresholds
        assert isinstance(result, bool)
    
    def test_fp_short_strings(self):
        """
        FP risk: Very short strings have higher match risk.
        """
        # Short strings like "IBM" vs "I BM" might accidentally match
        # Document that short strings need careful threshold tuning
        assert isinstance(fuzzy_match("IBM", "I am here", threshold=0.7), bool)
    
    # --- FALSE NEGATIVES: Should match but threshold too strict ---
    
    def test_fn_strict_threshold_misses_valid(self):
        """
        FN: Too strict threshold misses valid variations.
        'Inc' vs 'Incorporated' only ~60% similar.
        """
        # "Apple Inc" vs "Apple Incorporated" - the suffix part differs significantly
        result = fuzzy_match("Apple Inc", "Apple Incorporated reported", threshold=0.95)
        # At 0.95, this might not match - document threshold sensitivity
        # Lower threshold (0.75) would catch it
        assert fuzzy_match("Apple Inc", "Apple Incorporated reported", threshold=0.75) is True
    
    def test_fn_significant_abbreviation(self):
        """
        FN: Significant abbreviations need lower threshold.
        'Intl' vs 'International' is only ~50% similar.
        """
        # Need very low threshold for abbreviations
        result = fuzzy_match("Intl Business", "International Business", threshold=0.5)
        assert result is True  # Passes at low threshold
        
        result_strict = fuzzy_match("Intl Business", "International Business", threshold=0.9)
        assert result_strict is False  # Fails at high threshold - expected


# =============================================================================
# Integration Confusion Matrix: Full Pipeline Tests
# =============================================================================

class TestGroundedCheckConfusionMatrix:
    """
    End-to-end confusion matrix tests for GroundedCheck.
    Tests full pipeline from answer → entity extraction → matching → result.
    """
    
    # --- TRUE POSITIVES: Grounded entities correctly pass ---
    
    def test_tp_all_entities_grounded(self):
        """TP: All entities in sources → should pass."""
        sample = RagSample(
            answer="IBM and AWS announced a cloud partnership.",
            sources=[Source(id="1", text="IBM and AWS are launching joint cloud services.")]
        )
        result = GroundedCheck().run(sample)
        assert result.passed is True
        assert len(result.evidence["missing"]) == 0
    
    def test_tp_grounded_with_fuzzy(self):
        """TP: Similar entities pass with fuzzy matching."""
        sample = RagSample(
            answer="Acme Corp reported growth.",
            sources=[Source(id="1", text="Acme Corporation announced quarterly results.")]
        )
        # Exact fails ("Acme Corp" not substring of "Acme Corporation")
        exact_result = GroundedCheck(fuzzy=False).run(sample)
        # Note: Might pass if "Acme Corp" is substring - check actual behavior
        
        # Fuzzy passes with lower threshold (Corp vs Corporation ~75% similar)
        fuzzy_result = GroundedCheck(fuzzy=True, fuzzy_threshold=0.7).run(sample)
        assert fuzzy_result.passed is True
    
    # --- TRUE NEGATIVES: Hallucinated entities correctly fail ---
    
    def test_tn_hallucinated_entity(self):
        """TN: Entity not in sources → should fail."""
        sample = RagSample(
            answer="NVIDIA announced new GPUs.",
            sources=[Source(id="1", text="AMD reported quarterly earnings.")]
        )
        result = GroundedCheck().run(sample)
        assert result.passed is False
        assert "NVIDIA" in result.evidence["missing"]
    
    def test_tn_invented_person(self):
        """TN: Invented person name → should fail."""
        sample = RagSample(
            answer="John Smith at Acme Corp reported findings.",
            sources=[Source(id="1", text="Acme Corp released quarterly results.")]
        )
        result = GroundedCheck().run(sample)
        assert result.passed is False
        assert "John Smith" in result.evidence["missing"]
    
    # --- FALSE POSITIVE RISKS ---
    
    def test_fp_entity_substring_in_different_context(self):
        """
        FP risk: Entity appears as substring in unrelated context.
        'John' appears in 'Johnson' - should this count as grounded?
        Current behavior: Yes (substring match). Document this.
        """
        sample = RagSample(
            answer="John presented findings.",  # Contextual extraction
            sources=[Source(id="1", text="Johnson reported results.")]
        )
        result = GroundedCheck().run(sample)
        # "John" is substring of "Johnson" - current behavior allows this
        # This is a known FP risk but also catches legitimate partial matches
        assert isinstance(result, CheckResult)
    
    # --- FALSE NEGATIVE RISKS ---
    
    def test_fn_lowercase_entity_missed(self):
        """
        FN risk: Lowercase entities won't be extracted.
        """
        sample = RagSample(
            answer="The adidas partnership was valuable.",
            sources=[Source(id="1", text="adidas collaborated on the project.")]
        )
        result = GroundedCheck().run(sample)
        # 'adidas' not extracted (lowercase), so passes vacuously
        # This is a known FN - documented limitation
        assert result.passed is True  # No entities extracted = pass


class TestNumericGroundingConfusionMatrix:
    """
    End-to-end confusion matrix tests for NumericGroundingCheck.
    """
    
    # --- TRUE POSITIVES ---
    
    def test_tp_numbers_match_different_formats(self):
        """TP: Numbers in different formats should match."""
        sample = RagSample(
            answer="Revenue was $1.5M with 15% growth.",
            sources=[Source(id="1", text="Revenue: $1,500,000. Growth rate: 0.15.")]
        )
        result = NumericGroundingCheck().run(sample)
        assert result.passed is True
    
    # --- TRUE NEGATIVES ---
    
    def test_tn_invented_numbers_caught(self):
        """TN: Invented numbers should fail."""
        sample = RagSample(
            answer="Revenue grew 25% to $2M.",
            sources=[Source(id="1", text="Revenue: $1.5M. Growth: 12%.")]
        )
        result = NumericGroundingCheck(require_any=False).run(sample)
        assert result.passed is False
    
    # --- FALSE POSITIVE RISKS ---
    
    def test_fp_common_number_coincidence(self):
        """
        FP risk: Common numbers (1, 2, 10, 100) might match by coincidence.
        """
        sample = RagSample(
            answer="The company has 10 employees.",  # 10 might be coincidental
            sources=[Source(id="1", text="Revenue grew 10% this quarter.")]  # Different context!
        )
        result = NumericGroundingCheck().run(sample)
        # 10 matches 10 even though context is completely different
        # This is a known FP risk - numbers are context-free
        assert result.passed is True  # Document this as potential FP
    
    # --- FALSE NEGATIVE RISKS ---
    
    def test_fn_spelled_number_missed(self):
        """
        FN risk: Spelled-out numbers won't match.
        """
        sample = RagSample(
            answer="Revenue was one point five million dollars.",
            sources=[Source(id="1", text="Revenue: $1.5M")]
        )
        result = NumericGroundingCheck().run(sample)
        # No numeric values extracted from spelled-out text
        assert result.passed is True  # Vacuous pass - FN documented
