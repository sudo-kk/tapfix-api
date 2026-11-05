import json
import pytest

from app.extraction import (
    _build_response_tree,
    _filter_contradictory_toggles,
    _get_action_polarity,
    _normalize_setting_name,
    _prefer_polarity_from_siis,
)
from app.schema import Action, actionCategory, StepGroup
from app.validators import validate_full_response


def test_normalize_setting_name():
    """Verify that toggle verbs, articles, and setting suffixes are cleanly stripped."""
    assert _normalize_setting_name("Enable Touch Sensitivity") == "touch sensitivity"
    assert _normalize_setting_name("Disable Touch Sensitivity") == "touch sensitivity"
    assert _normalize_setting_name("Turn on Touch Sensitivity") == "touch sensitivity"
    assert _normalize_setting_name("Turn off Touch Sensitivity") == "touch sensitivity"
    assert _normalize_setting_name("Turn on the Touch Sensitivity Setting") == "touch sensitivity"
    assert _normalize_setting_name("Enable Bluetooth") == "bluetooth"
    assert _normalize_setting_name("Disable Bluetooth") == "bluetooth"
    assert _normalize_setting_name("Disable Full Screen Gestures") == "full screen gestures"
    assert _normalize_setting_name("Turn on Airplane Mode") == "airplane"
    assert _normalize_setting_name("Turn off Airplane Mode") == "airplane"


def test_get_action_polarity():
    """Verify polarity detection from action name and steps."""
    act_on = Action(
        actionName="Enable Touch Sensitivity",
        description="It will increase screen touch sensitivity",
        stepGroups=[
            StepGroup(
                steps=["Open Settings.", "Tap Display.", "Tap switch to turn it on."],
                actionableDeeplink=None,
                validationDeeplink=None,
            )
        ],
        category=actionCategory.auto,
    )
    act_off = Action(
        actionName="Disable Touch Sensitivity",
        description="It will reduce screen touch sensitivity",
        stepGroups=[
            StepGroup(
                steps=["Open Settings.", "Tap Display.", "Tap switch to turn it off."],
                actionableDeeplink=None,
                validationDeeplink=None,
            )
        ],
        category=actionCategory.auto,
    )
    act_none = Action(
        actionName="Remove Screen Protector",
        description="It will clean the device screen",
        stepGroups=[
            StepGroup(
                steps=["Peel off the screen protector.", "Wipe screen with cloth."],
                actionableDeeplink=None,
                validationDeeplink=None,
            )
        ],
        category=actionCategory.manual,
    )

    assert _get_action_polarity(act_on) == "on"
    assert _get_action_polarity(act_off) == "off"
    assert _get_action_polarity(act_none) is None


def test_prefer_polarity_from_siis():
    """Verify SIIS polarity preference extraction."""
    siis_on = "If your Bluetooth device fails to pair, you can enable Bluetooth in Settings."
    assert _prefer_polarity_from_siis("bluetooth", siis_on) == "on"

    siis_off = "To conserve battery or avoid interference, disable Bluetooth in Settings."
    assert _prefer_polarity_from_siis("bluetooth", siis_off) == "off"

    # Mentions both: ambiguous / conditional
    siis_both = (
        "Try enabling the Touch sensitivity option. If not using a protector, "
        "disable Touch sensitivity to prevent overly sensitive inputs."
    )
    assert _prefer_polarity_from_siis("touch sensitivity", siis_both) is None

    # Empty or unrelated SIIS
    assert _prefer_polarity_from_siis("touch sensitivity", "") is None
    assert _prefer_polarity_from_siis("touch sensitivity", "Unrelated camera troubleshooting.") is None


# ==============================================================================
# 1. Exact row_21 scenario: given payload with both Enable and Disable Touch Sensitivity
# ==============================================================================

def test_row21_touch_lag_filters_contradictory_touch_sensitivity():
    """Requirement 1:
    The exact row_21 touch-lag scenario: given the same query/SIIS content that previously
    produced both 'Enable Touch Sensitivity' and 'Disable Touch Sensitivity', the final response
    contains only ONE of them.
    """
    raw_row21_payload = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Touchscreen Responsiveness Troubleshooting",
                "title": "Touchscreen responsiveness issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Enable Touch Sensitivity",
                        "description": "It will increase screen touch input sensitivity",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Display.",
                                    "Tap the switch next to Touch sensitivity to turn it on.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Disable Full Screen Gestures",
                        "description": "It will switch navigation to standard buttons",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Display.",
                                    "Tap Navigation bar.",
                                    "Select Buttons.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Disable Touch Sensitivity",
                        "description": "It will reduce screen input touch sensitivity",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Display.",
                                    "Tap the switch next to Touch sensitivity to turn it off.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Remove Accessories and Clean Screen",
                        "description": "It will improve physical touchscreen input performance",
                        "category": "manual",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Remove any third-party screen protectors or accessories.",
                                    "Gently wipe the screen using a soft microfiber cloth.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Restart Device",
                        "description": "It will refresh device memory and system",
                        "category": "manual",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Swipe down from the top of the screen.",
                                    "Tap Power icon.",
                                    "Tap Restart.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Factory Data Reset",
                        "description": "It will reset device to factory settings",
                        "category": "critical",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Navigate to Settings.",
                                    "Tap General management.",
                                    "Tap Reset.",
                                    "Tap Factory data reset.",
                                ]
                            }
                        ],
                    },
                ],
            }
        ]
    }

    # Pass through _build_response_tree
    resp = _build_response_tree(raw_row21_payload)
    actions = resp.contexts[0].actions
    action_names = [a.actionName for a in actions]

    # Verify only ONE touch sensitivity action is present
    touch_sensitivity_actions = [
        name for name in action_names if "touch sensitivity" in name.lower()
    ]
    assert len(touch_sensitivity_actions) == 1, (
        f"Expected exactly 1 touch sensitivity action, found {touch_sensitivity_actions}"
    )
    assert touch_sensitivity_actions[0] == "Enable Touch Sensitivity"
    assert "Disable Touch Sensitivity" not in action_names

    # Verify Disable Full Screen Gestures is preserved
    assert "Disable Full Screen Gestures" in action_names

    # Verify overall schema validation succeeds
    is_valid, failures = validate_full_response(resp)
    assert is_valid, f"Validation failed: {failures}"


# ==============================================================================
# 2. Synthetic clean single-direction toggle: unaffected
# ==============================================================================

def test_single_direction_toggle_unaffected():
    """Requirement 2:
    A synthetic case with a clean single-direction toggle (e.g. 'Enable Bluetooth' alone)
    is unaffected — the guard must not remove a legitimate single toggle action.
    """
    raw_payload = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Bluetooth Connection Troubleshooting",
                "title": "Bluetooth pairing issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Enable Bluetooth",
                        "description": "It will turn on bluetooth wireless connection",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Connections.",
                                    "Tap Bluetooth to turn it on.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Restart Device",
                        "description": "It will refresh device memory and system",
                        "category": "manual",
                        "stepGroups": [
                            {"steps": ["Press and hold Power button.", "Tap Restart."]}
                        ],
                    },
                ],
            }
        ]
    }

    resp = _build_response_tree(raw_payload)
    actions = resp.contexts[0].actions
    action_names = [a.actionName for a in actions]

    assert "Enable Bluetooth" in action_names
    assert len(actions) == 2
    # Verify properties
    bt_action = next(a for a in actions if a.actionName == "Enable Bluetooth")
    assert bt_action.category == actionCategory.auto
    assert bt_action.stepGroups[0].actionableDeeplink is not None

    is_valid, failures = validate_full_response(resp)
    assert is_valid, f"Validation failed: {failures}"


# ==============================================================================
# 3. Two DIFFERENT settings toggled in the same plan: unaffected
# ==============================================================================

def test_two_different_settings_toggled_unaffected():
    """Requirement 3:
    A synthetic case with two DIFFERENT settings toggled in the same plan
    (e.g. 'Enable Bluetooth' AND 'Disable Full Screen Gestures') is unaffected —
    these are not a contradictory pair, both should remain.
    """
    raw_payload = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Settings Troubleshooting",
                "title": "Display controls issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Enable Bluetooth",
                        "description": "It will turn on bluetooth wireless connection",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Connections.",
                                    "Tap Bluetooth to turn it on.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Disable Full Screen Gestures",
                        "description": "It will switch navigation to standard buttons",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Display.",
                                    "Tap Navigation bar.",
                                    "Select Buttons.",
                                ]
                            }
                        ],
                    },
                    {
                        "actionName": "Inspect Physical Screen",
                        "description": "It will inspect screen for physical damage",
                        "category": "manual",
                        "stepGroups": [
                            {"steps": ["Check screen for cracks.", "Remove protective cases."]}
                        ],
                    },
                ],
            }
        ]
    }

    resp = _build_response_tree(raw_payload)
    actions = resp.contexts[0].actions
    action_names = [a.actionName for a in actions]

    # Both toggles for different settings must remain
    assert "Enable Bluetooth" in action_names
    assert "Disable Full Screen Gestures" in action_names
    assert "Inspect Physical Screen" in action_names
    assert len(actions) == 3

    is_valid, failures = validate_full_response(resp)
    assert is_valid, f"Validation failed: {failures}"


# ==============================================================================
# 4. Phase 5.1 and Phase 5.2 behavior unchanged
# ==============================================================================

def test_phase51_and_phase52_behavior_unchanged():
    """Requirement 4:
    Confirm Phase 5.1 (manual/null fallback) and Phase 5.2 (sequencing)
    behavior is unchanged for all other action types.
    """
    raw_payload = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this System Cache Troubleshooting",
                "title": "System cache issue",
                "score": 0.95,
                "actions": [
                    # Disruptive manual action generated first
                    {
                        "actionName": "Force Restart Device",
                        "description": "It will force restart your mobile device",
                        "category": "manual",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Press and hold Power and Volume down buttons for 10 seconds."
                                ]
                            }
                        ],
                    },
                    # Software action without catalog match -> Phase 5.1 downgrades to manual null
                    {
                        "actionName": "Clear App Cache and Data",
                        "description": "It will clear application cache data storage",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Navigate to Settings.",
                                    "Tap Apps.",
                                    "Tap Clear cache.",
                                    "Tap Clear data.",
                                ]
                            }
                        ],
                    },
                    # Genuine auto action
                    {
                        "actionName": "Enable Bluetooth",
                        "description": "It will turn on bluetooth wireless connection",
                        "category": "auto",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Open Settings.",
                                    "Tap Connections.",
                                    "Tap Bluetooth to turn it on.",
                                ]
                            }
                        ],
                    },
                    # Critical action
                    {
                        "actionName": "Factory Data Reset",
                        "description": "It will reset device to factory settings",
                        "category": "critical",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Navigate to Settings.",
                                    "Tap General management.",
                                    "Tap Reset.",
                                    "Tap Factory data reset.",
                                ]
                            }
                        ],
                    },
                    # Ordinary manual action
                    {
                        "actionName": "Inspect Hardware Damage",
                        "description": "It will inspect device for physical damage",
                        "category": "manual",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Remove cases from device.",
                                    "Check for cracks and corrosion.",
                                ]
                            }
                        ],
                    },
                ],
            }
        ]
    }

    resp = _build_response_tree(raw_payload)
    actions = resp.contexts[0].actions

    # Check Phase 5.1 fallback: "Clear App Cache and Data" must be downgraded to manual with null deeplinks
    cache_action = next(a for a in actions if a.actionName == "Clear App Cache and Data")
    assert cache_action.category == actionCategory.manual
    assert cache_action.stepGroups[0].actionableDeeplink is None
    assert cache_action.stepGroups[0].validationDeeplink is None

    # Check Phase 5.2 sequencing:
    # 0: auto (Enable Bluetooth)
    # 1: ordinary manual (Clear App Cache and Data, Inspect Hardware Damage)
    # 2: disruptive manual (Force Restart Device)
    # 3: critical (Factory Data Reset)
    expected_order = [
        "Enable Bluetooth",
        "Clear App Cache and Data",
        "Inspect Hardware Damage",
        "Force Restart Device",
        "Factory Data Reset",
    ]
    actual_order = [a.actionName for a in actions]
    assert actual_order == expected_order, f"Order mismatch: {actual_order} != {expected_order}"

    is_valid, failures = validate_full_response(resp)
    assert is_valid, f"Validation failed: {failures}"


def test_siis_direction_preference_replaces_earlier_action():
    """Verify that when SIIS explicitly indicates one direction, that direction is chosen even if generated second."""
    actions = [
        Action(
            actionName="Disable Bluetooth",
            description="It will turn off bluetooth connection",
            stepGroups=[
                StepGroup(
                    steps=["Open Settings.", "Tap Connections.", "Turn Bluetooth off."],
                    actionableDeeplink=None,
                    validationDeeplink=None,
                )
            ],
            category=actionCategory.auto,
        ),
        Action(
            actionName="Enable Bluetooth",
            description="It will turn on bluetooth connection",
            stepGroups=[
                StepGroup(
                    steps=["Open Settings.", "Tap Connections.", "Turn Bluetooth on."],
                    actionableDeeplink=None,
                    validationDeeplink=None,
                )
            ],
            category=actionCategory.auto,
        ),
    ]

    siis_content = "To fix audio issues, please enable Bluetooth in Settings."
    filtered = _filter_contradictory_toggles(actions, siis_content=siis_content)

    assert len(filtered) == 1
    assert filtered[0].actionName == "Enable Bluetooth"
