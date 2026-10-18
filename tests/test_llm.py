from axiom.llm import get_llm_provider, RulesBasedProvider, LLMSuggestion


def test_rules_based_provider():
    provider = RulesBasedProvider()

    record = {
        "workload_id": "matmul",
        "environment": {
            "hardware": {"architecture": "arm64", "physical_cores": 10, "cpu_brand": "Apple M4"}
        },
        "comparisons": [
            {"candidate_id": "interchanged", "speedup": 9.5, "correctness_passed": True}
        ]
    }

    # Should suggest multithreaded and neon since they are not in comparisons
    suggestions = provider.suggest_next_experiments(record)
    assert len(suggestions) >= 1
    assert any(s.suggested_candidate == "multithreaded" for s in suggestions)
    assert all(isinstance(s, LLMSuggestion) for s in suggestions)
    assert all("hypotheses only" in s.disclaimer for s in suggestions)

    # Explanation synthesis
    explanation = provider.explain_report_findings(record)
    assert "matmul" in explanation
    assert "interchanged" in explanation


def test_get_llm_provider_fallback():
    provider = get_llm_provider()
    assert isinstance(provider, RulesBasedProvider)
