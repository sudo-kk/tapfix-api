import json
import pytest
from unittest.mock import MagicMock, patch
from app.extraction import _parse_llm_json, extract_goal


def test_parse_llm_json_single_element_list_normalized():
    """A single-element list containing one dict is safely normalized."""
    raw = '[{"contexts": []}]'
    parsed = _parse_llm_json(raw)
    assert isinstance(parsed, dict)
    assert parsed == {"contexts": []}


def test_parse_llm_json_multi_element_list_rejected():
    """Multi-element lists must raise ValueError, not silently become valid."""
    raw = '[{"goal": "A"}, {"goal": "B"}]'
    with pytest.raises(ValueError, match="Malformed LLM JSON: expected dict or single-dict list"):
        _parse_llm_json(raw)


def test_parse_llm_json_empty_list_rejected():
    """Empty list must raise ValueError, not silently become valid."""
    raw = '[]'
    with pytest.raises(ValueError, match="Malformed LLM JSON: expected dict or single-dict list"):
        _parse_llm_json(raw)


@pytest.mark.parametrize("invalid_raw", [
    '"just a plain string"',
    '12345',
    'true',
    'null',
    '3.14159',
])
def test_parse_llm_json_non_dict_roots_rejected(invalid_raw):
    """Non-dict roots must raise ValueError, not silently become valid empty contexts."""
    with pytest.raises(ValueError, match="Malformed LLM JSON: expected dict"):
        _parse_llm_json(invalid_raw)


def test_parse_llm_json_malformed_syntax_rejected():
    """Malformed JSON syntax must raise JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_json('{"contexts": [}')


def test_malformed_gemini_output_cannot_become_valid_empty_response():
    """Prove that malformed Gemini output goes through retry and fails if uncorrected,
    never silently becoming an apparently valid empty response.
    """
    mock_client = MagicMock()
    mock_resp_malformed = MagicMock()
    mock_resp_malformed.text = '[{"bad": 1}, {"bad": 2}]'
    mock_client.models.generate_content.return_value = mock_resp_malformed

    with patch("app.extraction._get_client", return_value=mock_client):
        result = extract_goal("query", "title", "content")

    # Must have attempted retry
    assert result["_meta"]["retry_performed"] is True
    assert result["_meta"]["first_attempt_valid"] is False
    # Must NOT be marked as valid
    assert result["_meta"]["final_valid"] is False
    assert result["fallback"] == "validation_failed"
    assert any("JSON parsing/structure error" in f for f in result["_meta"]["failures"])


def test_malformed_gemini_output_recovers_on_retry():
    """Prove that malformed output on attempt 1 triggers retry and succeeds if attempt 2 is valid."""
    mock_client = MagicMock()
    mock_resp_1 = MagicMock()
    mock_resp_1.text = '"non-dict string response"'
    
    mock_resp_2 = MagicMock()
    mock_resp_2.text = json.dumps({"contexts": []})
    
    mock_client.models.generate_content.side_effect = [mock_resp_1, mock_resp_2]

    with patch("app.extraction._get_client", return_value=mock_client):
        result = extract_goal("query", "title", "content")

    assert result["_meta"]["retry_performed"] is True
    assert result["_meta"]["first_attempt_valid"] is False
    assert result["_meta"]["final_valid"] is True
    assert result["contexts"] == []
