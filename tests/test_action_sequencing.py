import pytest
from app.extraction import _build_response_tree, is_disruptive_action, _action_sort_key
from app.schema import Action, StepGroup, actionCategory


def _make_action(name: str, cat: str, steps: list[str], desc: str = "It will perform troubleshooting on your device") -> dict:
    return {
        "actionName": name,
        "description": desc,
        "category": cat,
        "stepGroups": [{"steps": steps}],
    }


def _make_response_payload(actions: list[dict]) -> dict:
    return {
        "contexts": [
            {
                "goal": "Follow these steps to perform troubleshooting",
                "title": "Device issue",
                "score": 0.95,
                "actions": actions,
            }
        ]
    }


def test_mixed_categories_ordered_auto_manual_critical():
    """1. Mixed categories are ordered: auto -> manual -> critical."""
    raw = _make_response_payload([
        _make_action("Perform Factory Data Reset", "critical", [
            "Navigate to and open Settings.",
            "Tap General management.",
            "Tap Reset.",
            "Tap Factory data reset.",
            "Swipe to and tap Reset.",
            "Tap Delete all.",
        ]),
        _make_action("Inspect Physical Hardware", "manual", [
            "Inspect the device, charger, and cable for damage.",
        ]),
        _make_action("Enable Bluetooth", "auto", [
            "Navigate to and open Settings.",
            "Tap Connections.",
            "Tap Bluetooth.",
            "Tap the switch to turn on Bluetooth.",
        ]),
    ])

    resp = _build_response_tree(raw)
    ordered_actions = resp.contexts[0].actions
    assert len(ordered_actions) == 3

    assert ordered_actions[0].actionName == "Enable Bluetooth"
    assert ordered_actions[0].category == actionCategory.auto

    assert ordered_actions[1].actionName == "Inspect Physical Hardware"
    assert ordered_actions[1].category == actionCategory.manual

    assert ordered_actions[2].actionName == "Perform Factory Data Reset"
    assert ordered_actions[2].category == actionCategory.critical


def test_stability_original_order_preserved_within_same_category():
    """2. Original order is preserved within the same category."""
    raw = _make_response_payload([
        _make_action("Enable Bluetooth", "auto", [
            "Navigate to Settings. Tap Connections. Tap Bluetooth.",
        ]),
        _make_action("Inspect Physical Hardware", "manual", [
            "Inspect the device for damage.",
        ]),
        _make_action("Disable Touch Sensitivity", "auto", [
            "Navigate to Settings. Tap Display. Tap switch next to Touch sensitivity to disable it.",
        ]),
        _make_action("Charge The Device", "manual", [
            "Connect the device to a charger for at least one hour.",
        ]),
    ])

    resp = _build_response_tree(raw)
    names = [a.actionName for a in resp.contexts[0].actions]

    # Auto actions stay in original relative order: Bluetooth before Touch Sensitivity
    assert names.index("Enable Bluetooth") < names.index("Disable Touch Sensitivity")
    # Manual actions stay in original relative order: Inspect before Charge
    assert names.index("Inspect Physical Hardware") < names.index("Charge The Device")
    # All auto come before all manual
    assert names == [
        "Enable Bluetooth",
        "Disable Touch Sensitivity",
        "Inspect Physical Hardware",
        "Charge The Device",
    ]


def test_disruptive_manual_action_restart_moved_after_ordinary_manual():
    """3. A disruptive manual action such as 'Restart Device' is moved after ordinary manual actions."""
    raw = _make_response_payload([
        _make_action("Restart Device", "manual", [
            "Press and hold the Power button, then tap Restart.",
        ]),
        _make_action("Inspect Physical Hardware", "manual", [
            "Inspect the device, charger, and cable for damage.",
        ]),
        _make_action("Charge The Device", "manual", [
            "Allow the device to charge for at least one hour.",
        ]),
    ])

    resp = _build_response_tree(raw)
    names = [a.actionName for a in resp.contexts[0].actions]

    assert names == [
        "Inspect Physical Hardware",
        "Charge The Device",
        "Restart Device",
    ]


def test_disruptive_manual_action_safe_mode_moved_to_end():
    """4. 'Restart in Safe Mode' is also moved to the end of manual actions."""
    raw = _make_response_payload([
        _make_action("Restart in Safe Mode", "manual", [
            "Touch and hold Power off. Tap Safe mode.",
        ]),
        _make_action("Clean Screen and Ports", "manual", [
            "Gently wipe screen with microfiber cloth.",
        ]),
        _make_action("Check Charger", "manual", [
            "Inspect charger for bent pins.",
        ]),
    ])

    resp = _build_response_tree(raw)
    names = [a.actionName for a in resp.contexts[0].actions]

    assert names == [
        "Clean Screen and Ports",
        "Check Charger",
        "Restart in Safe Mode",
    ]


def test_physical_action_force_restart_remains_manual_null():
    """5. A physical manual action such as 'Force Restart Device' remains manual + null and is not converted to critical."""
    raw = _make_response_payload([
        _make_action("Force Restart Device", "manual", [
            "Press and hold both Power and Volume down buttons for 20 seconds.",
        ]),
    ])

    resp = _build_response_tree(raw)
    action = resp.contexts[0].actions[0]

    assert action.actionName == "Force Restart Device"
    assert action.category == actionCategory.manual
    assert action.stepGroups[0].actionableDeeplink is None
    assert action.stepGroups[0].validationDeeplink is None


def test_known_auto_action_bluetooth_remains_auto_with_deeplink():
    """6. A known auto action such as 'Enable Bluetooth' remains auto with its existing real catalog deeplink."""
    raw = _make_response_payload([
        _make_action("Enable Bluetooth", "auto", [
            "Navigate to and open Settings.",
            "Tap Connections.",
            "Tap Bluetooth.",
            "Tap the switch to turn on Bluetooth.",
        ]),
    ])

    resp = _build_response_tree(raw)
    action = resp.contexts[0].actions[0]

    assert action.category == actionCategory.auto
    sg = action.stepGroups[0]
    assert sg.actionableDeeplink is not None
    assert sg.actionableDeeplink.deeplink.startswith("bixby://masked/act/")
    assert sg.actionableDeeplink.deeplink != "bixby://dummy_positive"


def test_factory_data_reset_remains_critical_at_end():
    """7. Factory Data Reset remains critical and stays at the end."""
    raw = _make_response_payload([
        _make_action("Perform Factory Data Reset", "critical", [
            "Navigate to Settings. Tap Reset. Tap Factory data reset.",
        ]),
        _make_action("Restart Device", "manual", [
            "Tap Restart to reboot device.",
        ]),
        _make_action("Inspect Hardware", "manual", [
            "Inspect physical hardware.",
        ]),
        _make_action("Enable Bluetooth", "auto", [
            "Navigate to Settings. Tap Connections. Tap Bluetooth.",
        ]),
    ])

    resp = _build_response_tree(raw)
    names = [a.actionName for a in resp.contexts[0].actions]

    # auto -> ordinary manual -> disruptive manual -> critical
    assert names == [
        "Enable Bluetooth",
        "Inspect Hardware",
        "Restart Device",
        "Perform Factory Data Reset",
    ]
    assert resp.contexts[0].actions[-1].category == actionCategory.critical


def test_phase51_fallback_unmatched_remains_manual_null():
    """8. Existing Phase 5.1 fallback behavior remains unchanged: unmatched action -> manual + null."""
    raw = _make_response_payload([
        _make_action("Clear App Cache and Data", "auto", [
            "Navigate to Settings. Tap Apps. Tap Storage. Tap Clear cache. Tap Clear data.",
        ]),
    ])

    resp = _build_response_tree(raw)
    action = resp.contexts[0].actions[0]

    assert action.category == actionCategory.manual
    assert action.stepGroups[0].actionableDeeplink is None
    assert action.stepGroups[0].validationDeeplink is None


def test_multi_stepgroup_safety_partial_match_downgrades_to_manual():
    """Edge case: multi-StepGroup software action where StepGroup 1 matches but StepGroup 2 fails.
    The action must NOT remain auto/critical solely because StepGroup 1 matched; it must become manual
    with null deeplinks per Phase 5.1.
    """
    raw = {
        "contexts": [
            {
                "goal": "Follow these steps to perform troubleshooting",
                "title": "Device issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Configure Wireless And Clear Cache",
                        "description": "It will configure wireless connection and clear cache",
                        "category": "auto",
                        "stepGroups": [
                            {
                                # StepGroup 1 has a real catalog match (Bluetooth)
                                "steps": [
                                    "Navigate to and open Settings.",
                                    "Tap Connections.",
                                    "Tap Bluetooth.",
                                    "Tap the switch to turn on Bluetooth.",
                                ]
                            },
                            {
                                # StepGroup 2 has NO catalog match
                                "steps": [
                                    "Navigate to and open Settings.",
                                    "Tap Apps.",
                                    "Select email app.",
                                    "Tap Storage.",
                                    "Tap Clear cache.",
                                    "Tap Clear data, and then tap OK.",
                                ]
                            },
                        ],
                    }
                ],
            }
        ]
    }

    resp = _build_response_tree(raw)
    action = resp.contexts[0].actions[0]

    # Action MUST NOT remain auto
    assert action.category == actionCategory.manual
    assert len(action.stepGroups) == 2
    # All StepGroups must be null in a manual action
    assert action.stepGroups[0].actionableDeeplink is None
    assert action.stepGroups[0].validationDeeplink is None
    assert action.stepGroups[1].actionableDeeplink is None
    assert action.stepGroups[1].validationDeeplink is None


@pytest.mark.parametrize(
    "name,desc,expected",
    [
        ("Restart Device", "It will restart your mobile device", True),
        ("Reboot Device", "It will reboot your mobile device", True),
        ("Reboot into Safe Mode", "It will boot phone into safe mode", True),
        ("Power cycle your phone", "It will power cycle your mobile phone", True),
        ("Power-cycle the handset", "It will power-cycle device to clear state", True),
        ("Force Restart Device", "It will force restart your mobile device", True),
        ("Test in Safe Mode", "It will isolate issues in safe mode", True),
        ("Restart in Safe Mode", "It will reboot the phone into safe mode", True),
        ("Inspect Physical Hardware", "It will inspect device for physical damage", False),
        ("Charge Device", "It will charge your mobile device battery", False),
        ("Connect External Accessories", "It will connect physical mouse and keyboard", False),
        ("Visit Service Center", "It will service your damaged device", False),
        ("Contact Samsung Support", "It will request technical service or repair", False),
        ("Remove Screen Protector", "It will clean screen and remove accessories", False),
        ("Adjust Touch Sensitivity", "It will adjust touchscreen sensitivity settings", False),
        ("Disable Gesture Navigation", "It will switch navigation to button mode", False),
    ],
)
def test_disruptive_action_detection_paraphrases(name, desc, expected):
    """Test disruptive action detection covering paraphrases and varied wording."""
    act = Action(
        actionName=name,
        description=desc,
        stepGroups=[StepGroup(steps=["Sample step."])],
        category=actionCategory.manual,
    )
    assert is_disruptive_action(act) == expected
