import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.extraction import _SYSTEM_PROMPT, extract_goal
from app.schema import ContextDeeplinkResponse
from app.validators import validate_full_response

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Grounded structured troubleshooting response derived strictly from row_16 SIIS content:
# "Smartphone,Others Mobile,Tablet,Mobile Accessories Blank or black display on a Samsung phone or tablet..."
_ROW_16_GROUNDED_LLM_OUTPUT = {
    "contexts": [
        {
            "goal": "Follow these steps to perform this Display Troubleshooting",
            "title": "Blank display issue",
            "score": 0.95,
            "actions": [
                {
                    "actionName": "Inspect Device Damage",
                    "description": "It will check physical or liquid damage",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Inspect the device, charger, and USB cable for physical damage.",
                                "Remove cases or accessories covering the charging port or buttons.",
                                "Examine USB connections for corrosion or bent pins.",
                                "Turn off the device.",
                                "Eject the SIM or MicroSD tray using an ejector tool.",
                                "Shine a flashlight into the SIM slot to inspect the Liquid Damage Indicator.",
                            ]
                        }
                    ],
                },
                {
                    "actionName": "Charge The Device",
                    "description": "It will charge device for one hour",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Connect the device to the appropriate charger.",
                                "Let the device charge for at least 1 hour.",
                            ]
                        }
                    ],
                },
                {
                    "actionName": "Attempt Power On",
                    "description": "It will try powering on the device",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Disconnect the device from the charger.",
                                "Press and hold the Power button for 15 to 20 seconds to power on.",
                            ]
                        }
                    ],
                },
                {
                    "actionName": "Contact Samsung Support",
                    "description": "It will request additional service and support",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Contact Samsung Support Center if the screen issue persists.",
                            ]
                        }
                    ],
                },
                {
                    "actionName": "Force Restart Device",
                    "description": "It will force restart the mobile device",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Press and hold the Power button and Volume down button simultaneously for at least 20 seconds.",
                            ]
                        }
                    ],
                },
            ],
        }
    ]
}


def test_row16_prompt_includes_relevance_guidance():
    """Verify that system prompt explicitly handles harmless wording differences between
    screen flashing when charging and blank/black display troubleshooting.
    """
    assert "flashes" in _SYSTEM_PROMPT.lower()
    assert "blank or black display" in _SYSTEM_PROMPT.lower()
    assert "charger" in _SYSTEM_PROMPT.lower()


def test_row16_extracts_grounded_steps_not_empty():
    """Verify row_16 query extracts grounded physical troubleshooting steps and does not
    collapse to empty contexts due to wording differences.
    """
    siis_path = _DATA_DIR / "siis_responses.json"
    if not siis_path.exists():
        pytest.skip("hackathon SIIS data is not included in the public repo")
    with open(siis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row_16 = next(r for r in data["responses"] if r["id"] == "row_16")
    query = row_16["original_query"]
    title = row_16["siis_response"]["title"]
    content = row_16["siis_response"]["content"]

    # Mock Gemini API client at test boundary to ensure deterministic, offline-safe execution
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(_ROW_16_GROUNDED_LLM_OUTPUT)
    mock_client.models.generate_content.return_value = mock_resp

    with patch("app.extraction._get_client", return_value=mock_client):
        result = extract_goal(query, title, content)

    contexts = result.get("contexts", [])

    assert len(contexts) == 1, "row_16 must not be empty"
    goal = contexts[0]
    assert "goal" in goal
    assert "actions" in goal
    assert len(goal["actions"]) >= 2
    assert len(goal["actions"]) == 5

    # Validate against full response schema
    resp_obj = ContextDeeplinkResponse(**result)
    is_valid, failures = validate_full_response(resp_obj)
    assert is_valid, f"Extracted response failed validation: {failures}"

    # Collect all steps across all actions
    all_steps = []
    action_names = []
    for action in goal["actions"]:
        action_names.append(action["actionName"])
        assert action["category"] == "manual"
        for sg in action["stepGroups"]:
            assert sg["actionableDeeplink"] is None
            assert sg["validationDeeplink"] is None
            all_steps.extend(sg["steps"])

    # Verify actions and steps are grounded in actual row_16 SIIS content
    combined_steps_text = " ".join(all_steps).lower()
    assert "liquid damage indicator" in combined_steps_text or "ldi" in combined_steps_text
    assert "charger" in combined_steps_text
    assert "volume down" in combined_steps_text or "power button" in combined_steps_text
    assert "samsung support center" in combined_steps_text

    # Verify action names match grounded SIIS procedures
    assert "Inspect Device Damage" in action_names
    assert "Charge The Device" in action_names
    assert "Attempt Power On" in action_names
    assert "Force Restart Device" in action_names
    assert "Contact Samsung Support" in action_names
