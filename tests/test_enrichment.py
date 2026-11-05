import json
from unittest.mock import MagicMock
import pytest

from app.enrichment import (
    _normalize_for_comparison,
    _generate_deterministic_fallbacks,
    generate_paraphrases,
)


def _mock_gemini_client(monkeypatch, return_texts):
    """Helper to mock Gemini client with a sequence or single return text."""
    if isinstance(return_texts, str):
        return_texts = [return_texts]

    call_index = {"val": 0}

    def mock_generate_content(*args, **kwargs):
        idx = call_index["val"]
        if idx < len(return_texts):
            text = return_texts[idx]
            call_index["val"] += 1
        else:
            text = return_texts[-1]

        if isinstance(text, Exception):
            raise text

        resp = MagicMock()
        resp.text = text
        return resp

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_generate_content
    monkeypatch.setattr("app.enrichment._get_genai_client", lambda: mock_client)
    return mock_client


def test_clean_gemini_variations_returns_exact_count(monkeypatch):
    """Test 1: 9 clean Gemini variations -> exactly 9 returned."""
    variations = [
        "How can I resolve my Galaxy S22 screen going blank?",
        "My Galaxy S22 display turns completely white and shows no content",
        "Screen stays totally black on Galaxy S22 during app launch",
        "Troubleshooting steps for blank screen problem on Galaxy S22",
        "Why does my Galaxy S22 display show nothing when using apps?",
        "Galaxy S22 phone screen is not displaying any graphics or text",
        "Assistance needed for blank white display on Samsung S22",
        "My Galaxy S22 screen is black plz help fix it",
        "Looking for instructions to fix blank display on Galaxy S22",
    ]
    _mock_gemini_client(monkeypatch, json.dumps(variations))

    query = "My Galaxy S22 screen turns completely blank or white and no text appears"
    result = generate_paraphrases(query, n=9)

    assert len(result) == 9
    assert result == variations


def test_gemini_duplicates_are_removed(monkeypatch):
    """Test 2: Gemini returns duplicates -> duplicates removed, retry/fallback reaches >=8."""
    # 9 items returned, but several are exact or case duplicates
    gemini_output_1 = json.dumps([
        "Display is blank on Galaxy S22",
        "Display is blank on Galaxy S22",  # Exact duplicate
        "display is blank on galaxy s22",  # Case duplicate
        "Screen goes black when launching apps",
        "How do I fix blank screen on Galaxy S22?",
    ])
    gemini_output_2 = json.dumps([
        "Galaxy S22 display completely unresponsive and black",
        "Samsung S22 screen issue troubleshooting",
        "What to do when Galaxy S22 screen goes white",
        "Steps to recover blank display on Galaxy S22",
    ])
    _mock_gemini_client(monkeypatch, [gemini_output_1, gemini_output_2])

    query = "My Galaxy S22 screen turns completely blank or white and no text appears"
    result = generate_paraphrases(query, n=9)

    assert 8 <= len(result) <= 10
    # Ensure all returned variations are unique
    normalized = [_normalize_for_comparison(r) for r in result]
    assert len(normalized) == len(set(normalized))


def test_original_query_removed_if_returned_by_gemini(monkeypatch):
    """Test 3: Gemini returns the original query -> original removed and replaced."""
    query = "My Galaxy S22 screen turns completely blank or white and no text appears"
    gemini_output = json.dumps([
        query,  # Exact original query!
        f"  {query}  ",  # Whitespace duplicate of original
        query.upper(),  # Case duplicate of original
        f"1. {query}",  # Numbered duplicate of original
        "Different variation 1 for Galaxy S22 blank display",
        "Different variation 2 for Galaxy S22 blank display",
        "Different variation 3 for Galaxy S22 blank display",
        "Different variation 4 for Galaxy S22 blank display",
        "Different variation 5 for Galaxy S22 blank display",
    ])
    _mock_gemini_client(monkeypatch, gemini_output)

    result = generate_paraphrases(query, n=9)

    assert 8 <= len(result) <= 10
    # Original query must NOT be in the results in any form
    norm_query = _normalize_for_comparison(query)
    for r in result:
        assert _normalize_for_comparison(r) != norm_query
        assert r.strip() != query.strip()


def test_gemini_fewer_than_8_reaches_minimum(monkeypatch):
    """Test 4: Gemini returns only 5 usable variations -> retry/fallback reaches >=8."""
    gemini_output = json.dumps([
        "Variation A for Galaxy S22 blank display",
        "Variation B for Galaxy S22 blank display",
        "Variation C for Galaxy S22 blank display",
        "Variation D for Galaxy S22 blank display",
        "Variation E for Galaxy S22 blank display",
    ])
    # Attempt 1 returns 5, retry returns empty list
    _mock_gemini_client(monkeypatch, [gemini_output, "[]"])

    query = "My Galaxy S22 screen turns completely blank or white and no text appears"
    result = generate_paraphrases(query, n=9)

    assert len(result) >= 8
    assert len(result) <= 10
    # All 5 from attempt 1 are present
    assert result[:5] == [
        "Variation A for Galaxy S22 blank display",
        "Variation B for Galaxy S22 blank display",
        "Variation C for Galaxy S22 blank display",
        "Variation D for Galaxy S22 blank display",
        "Variation E for Galaxy S22 blank display",
    ]
    # And remainder filled with unique fallbacks
    normalized = [_normalize_for_comparison(r) for r in result]
    assert len(normalized) == len(set(normalized))


def test_gemini_empty_response_fallback(monkeypatch):
    """Test 5: Gemini returns [] -> fallback produces >=8 unique variations."""
    _mock_gemini_client(monkeypatch, "[]")

    query = "My Samsung A115G tablet screen flashes and then goes completely blank"
    result = generate_paraphrases(query, n=9)

    assert 8 <= len(result) <= 10
    for r in result:
        assert isinstance(r, str)
        assert len(r) > 10
        # Device name preserved
        assert "Samsung A115G" in r or "tablet" in r.lower()

    normalized = [_normalize_for_comparison(r) for r in result]
    assert len(normalized) == len(set(normalized))


def test_gemini_exception_fallback(monkeypatch):
    """Test 6: Gemini throws an exception -> fallback produces >=8 unique variations."""
    _mock_gemini_client(monkeypatch, [Exception("Google API connection timeout: 504 Deadline Exceeded")])

    query = "My Galaxy S25 has a floating circle that constantly hovers on my screen"
    result = generate_paraphrases(query, n=9)

    assert 8 <= len(result) <= 10
    normalized = [_normalize_for_comparison(r) for r in result]
    assert len(normalized) == len(set(normalized))
    assert _normalize_for_comparison(query) not in normalized


def test_gemini_malformed_json_fallback(monkeypatch):
    """Test 7: Gemini returns malformed output -> fallback produces >=8 unique variations."""
    malformed_responses = [
        "This is not JSON at all, just arbitrary text.",
        "```json\n{invalid json syntax here\n```",
        json.dumps({"error": "Resource exhausted", "code": 429}),
    ]
    for bad_text in malformed_responses:
        _mock_gemini_client(monkeypatch, bad_text)
        query = "My Galaxy Z Flip 6 screen flickers and goes blank whenever I open it"
        result = generate_paraphrases(query, n=9)
        assert 8 <= len(result) <= 10
        normalized = [_normalize_for_comparison(r) for r in result]
        assert len(normalized) == len(set(normalized))


def test_case_and_whitespace_duplicates_deduplicated(monkeypatch):
    """Test 8: Case/whitespace duplicates -> strictly deduplicated."""
    gemini_output = json.dumps([
        "  Fix Galaxy S24 Ultra display issue  ",
        "fix galaxy s24 ultra display issue",
        "FIX GALAXY S24 ULTRA DISPLAY ISSUE.",
        "Fix Galaxy S24 Ultra Display Issue!",
        "Second valid distinct variation for Galaxy S24",
    ])
    _mock_gemini_client(monkeypatch, gemini_output)

    query = "My Galaxy S24 Ultra screen is completely black and won't turn on"
    result = generate_paraphrases(query, n=9)

    assert 8 <= len(result) <= 10
    normalized = [_normalize_for_comparison(r) for r in result]
    assert len(normalized) == len(set(normalized))
    # Count of the first variation root
    assert normalized.count("fix galaxy s24 ultra display issue") == 1


def test_result_count_capped_at_10(monkeypatch):
    """Test 9: Result count never exceeds 10 even if requested or returned."""
    # Gemini returns 15 distinct items
    many_variations = [f"Distinct variation number {i} for screen issue" for i in range(15)]
    _mock_gemini_client(monkeypatch, json.dumps(many_variations))

    query = "My Galaxy phone screen has visual distortions"
    result_n15 = generate_paraphrases(query, n=15)
    assert len(result_n15) == 10  # Capped at 10

    result_n10 = generate_paraphrases(query, n=10)
    assert len(result_n10) == 10

    result_n8 = generate_paraphrases(query, n=8)
    assert len(result_n8) == 8


def test_existing_enrichment_behavior_compatibility(monkeypatch):
    """Test 10: Existing enrichment behavior remains compatible."""
    gemini_output = json.dumps([
        f"Natural variation {i} for Samsung mobile issue" for i in range(1, 10)
    ])
    _mock_gemini_client(monkeypatch, gemini_output)

    query = "My phone screen flashes extremely quickly whenever I plug in a charger"
    result = generate_paraphrases(query)

    assert isinstance(result, list)
    assert len(result) == 9
    for r in result:
        assert isinstance(r, str)
        assert len(r) > 5
        # Ensure no URL leakage
        assert "http" not in r
        assert "www." not in r


def test_degenerate_input_handling_explicit():
    """Test 11: Degenerate input handling is controlled and explicit."""
    for degenerate in ["", "   ", "a", "no", "x y"]:
        with pytest.raises(ValueError) as exc_info:
            generate_paraphrases(degenerate)
        assert "degenerate" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
