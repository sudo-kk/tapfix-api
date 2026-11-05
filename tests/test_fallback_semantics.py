import pytest
from app.extraction import _build_response_tree
from app.schema import actionCategory


def _make_mock_llm_json(action_name: str, description: str, steps: list[str], category: str):
    return {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Device Troubleshooting",
                "title": "Device issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": action_name,
                        "description": description,
                        "category": category,
                        "stepGroups": [
                            {
                                "steps": steps,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_case_a_clear_app_cache_and_data_downgrades_to_manual_null():
    """Case A — Clear App Cache and Data has no confident catalog match.
    Even if LLM proposes category="auto", response assembly must override it to:
    - category = "manual"
    - actionableDeeplink = null
    - validationDeeplink = null
    """
    raw_llm = _make_mock_llm_json(
        action_name="Clear App Cache and Data",
        description="It will clear application cache and data",
        steps=[
            "Navigate to and open Settings.",
            "Tap Apps.",
            "Select your email app.",
            "Tap Storage.",
            "Tap Clear cache.",
            "Tap Clear data, and then tap OK.",
        ],
        category="auto",
    )

    resp = _build_response_tree(raw_llm)
    assert len(resp.contexts) == 1
    action = resp.contexts[0].actions[0]

    assert action.actionName == "Clear App Cache and Data"
    assert action.category == actionCategory.manual
    assert len(action.stepGroups) == 1
    sg = action.stepGroups[0]
    assert sg.actionableDeeplink is None
    assert sg.validationDeeplink is None


@pytest.mark.parametrize("action_name", ["Adjust Phone Aspect Ratio", "Change Phone Aspect Ratio"])
def test_case_b_aspect_ratio_downgrades_to_manual_null(action_name):
    """Case B — Adjust/Change Phone Aspect Ratio has no confident catalog match.
    Even if LLM proposes category="auto", response assembly must override it to:
    - category = "manual"
    - actionableDeeplink = null
    - validationDeeplink = null
    """
    raw_llm = _make_mock_llm_json(
        action_name=action_name,
        description="It will adjust phone aspect ratio settings",
        steps=[
            "Tap the Smart View icon on your phone.",
            "Tap More options (the three vertical dots).",
            "Tap Settings.",
            "Tap Phone aspect ratio.",
            "Select your desired aspect ratio.",
        ],
        category="auto",
    )

    resp = _build_response_tree(raw_llm)
    assert len(resp.contexts) == 1
    action = resp.contexts[0].actions[0]

    assert action.actionName == action_name
    assert action.category == actionCategory.manual
    assert len(action.stepGroups) == 1
    sg = action.stepGroups[0]
    assert sg.actionableDeeplink is None
    assert sg.validationDeeplink is None


def test_case_c_known_good_match_enable_bluetooth_remains_auto_with_real_deeplink():
    """Case C — Enable Bluetooth has a trustworthy catalog match.
    Response assembly must:
    - preserve category="auto"
    - provide a real catalog deeplink from data/deeplinks.json
    - NOT return dummy_positive
    """
    raw_llm = _make_mock_llm_json(
        action_name="Enable Bluetooth",
        description="It will turn on bluetooth wireless connection",
        steps=[
            "Navigate to and open Settings.",
            "Tap Connections.",
            "Tap Bluetooth.",
            "Tap the switch to turn on Bluetooth.",
        ],
        category="auto",
    )

    resp = _build_response_tree(raw_llm)
    assert len(resp.contexts) == 1
    action = resp.contexts[0].actions[0]

    assert action.actionName == "Enable Bluetooth"
    assert action.category == actionCategory.auto
    assert len(action.stepGroups) == 1
    sg = action.stepGroups[0]

    assert sg.actionableDeeplink is not None
    assert sg.actionableDeeplink.deeplink.startswith("bixby://masked/act/")
    assert sg.actionableDeeplink.deeplink != "bixby://dummy_positive"
    assert "bluetooth" in sg.actionableDeeplink.description.lower() or "bluetooth" in sg.actionableDeeplink.message.lower()


def test_case_d_existing_physical_manual_action_remains_manual_null():
    """Case D — Known physical/hardware action such as Force Restart Device.
    Response assembly must:
    - preserve category="manual"
    - set actionableDeeplink = null
    - set validationDeeplink = null
    """
    raw_llm = _make_mock_llm_json(
        action_name="Force Restart Device",
        description="It will force restart your mobile device",
        steps=[
            "Press and hold both the Power or Side button and the Volume down button simultaneously for at least 20 seconds.",
        ],
        category="manual",
    )

    resp = _build_response_tree(raw_llm)
    assert len(resp.contexts) == 1
    action = resp.contexts[0].actions[0]

    assert action.actionName == "Force Restart Device"
    assert action.category == actionCategory.manual
    assert len(action.stepGroups) == 1
    sg = action.stepGroups[0]
    assert sg.actionableDeeplink is None
    assert sg.validationDeeplink is None
