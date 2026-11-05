import json
from pathlib import Path
import pytest

from app.deeplink_match import match_deeplink

_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_output.json"


def test_top3_structure_and_keys():
    results = match_deeplink("Turn on Wi-Fi", top_k=3)
    assert len(results) == 3

    for r in results:
        assert "id" in r and r["id"].startswith("DL-")
        assert "deeplink" in r
        assert "description" in r
        assert "message" in r
        assert "originalType" in r
        assert "score" in r and isinstance(r["score"], float)
        assert r["score"] >= 0.5
        assert "validation" in r

    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_fallback_for_nonsense_input():
    nonsense = "asdfghjkl qwertyuiop zxcvbnm 1234567890 blorp foo"
    results = match_deeplink(nonsense)
    assert len(results) == 0


def test_fallback_for_empty_input():
    results = match_deeplink("")
    assert len(results) == 0


def test_fallback_for_short_input():
    query = "xyzzy"
    results = match_deeplink(query)
    assert len(results) == 0


def test_known_catalog_semantic_retrieval_5_cases():
    """5 catalog-grounded queries where expected IDs exist in data/deeplinks.json.

    At least 4 of 5 must return the correct deeplink ID in top-3 matches.
    """
    cases = [
        ("Turn on Wi-Fi", "DL-0492"),
        ("Adjust screen brightness", "DL-0496"),
        ("Enable adaptive brightness", "DL-0021"),
        ("Enable Bluetooth on the device", "DL-0495"),
        ("Open battery settings to check usage", "DL-0560"),
    ]

    matches = 0
    for query, expected_id in cases:
        results = match_deeplink(query, top_k=3)
        top_ids = [r["id"] for r in results]
        if expected_id in top_ids:
            matches += 1

    assert matches >= 4, f"Expected >= 4 matches, got {matches}/5"


def test_sample_output_steps_diagnostic():
    """Diagnostic probe of sample_output.json steps.

    sample_output.json contains only 1 actionable deeplink (DL-0542) for
    the entire group of steps 1-3, and manual steps (null deeplink) for steps 4-5.
    This diagnostic verifies valid retrieval for the auto action and no-match for the manual action.
    """
    with open(_SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample = json.load(f)

    # For the auto action "Back Up Phone Data", the combined step text should match DL-0542
    auto_action = sample["response"]["contexts"][0]["actions"][0]
    auto_steps = " ".join(auto_action["stepGroups"][0]["steps"])
    results = match_deeplink(auto_steps, top_k=3, action_name=auto_action["actionName"])
    assert len(results) >= 1
    assert results[0]["id"] == "DL-0542"
    assert results[0]["deeplink"] == "bixby://masked/act/b3ed3ed663"
    assert results[0]["deeplink"] != "bixby://dummy_positive"

    # For the manual action "Schedule Screen Repair Service", there is no catalog deeplink
    manual_action = sample["response"]["contexts"][0]["actions"][1]
    manual_steps = " ".join(manual_action["stepGroups"][0]["steps"])
    manual_results = match_deeplink(manual_steps, top_k=3, action_name=manual_action["actionName"])
    assert len(manual_results) == 0


def test_polarity_enable_touch_sensitivity():
    results = match_deeplink(
        "Navigate to and open Settings. Tap Display. Tap the switch next to Touch sensitivity to enable it.",
        top_k=3,
        action_name="Enable Touch Sensitivity",
    )
    assert len(results) >= 1
    top = results[0]
    assert top["id"] == "DL-0126"
    assert top["originalType"] == "onURL"
    assert "Enables touch sensitivity" in top["description"]
    # Verify offURL is not selected
    for r in results:
        assert r["id"] != "DL-0125"


def test_polarity_disable_touch_sensitivity():
    results = match_deeplink(
        "Navigate to and open Settings. Tap Display. Tap the switch next to Touch sensitivity to disable it.",
        top_k=3,
        action_name="Disable Touch Sensitivity",
    )
    assert len(results) >= 1
    top = results[0]
    assert top["id"] == "DL-0125"
    assert top["originalType"] == "offURL"
    assert "Disables touch sensitivity" in top["description"]
    # Verify onURL is not selected
    for r in results:
        assert r["id"] != "DL-0126"


def test_polarity_enable_bluetooth():
    results = match_deeplink(
        "Open Settings. Tap Connections. Tap Bluetooth. Tap the switch to turn it on.",
        top_k=3,
        action_name="Enable Bluetooth",
    )
    assert len(results) >= 1
    top = results[0]
    assert top["id"] == "DL-0495"
    assert top["originalType"] == "onURL"
    assert "Enables bluetooth" in top["description"]
    # Verify disable Bluetooth is not selected
    for r in results:
        assert r["id"] not in ("DL-0494", "DL-0042")


def test_polarity_disable_bluetooth():
    results = match_deeplink(
        "Open Settings. Tap Connections. Tap Bluetooth. Tap the switch to turn it off.",
        top_k=3,
        action_name="Disable Bluetooth",
    )
    assert len(results) >= 1
    top = results[0]
    assert top["id"] == "DL-0494"
    assert top["originalType"] == "offURL"
    assert "Disables bluetooth" in top["description"]
    # Verify enable Bluetooth is not selected
    for r in results:
        assert r["id"] not in ("DL-0495", "DL-0043")


def test_verify_internet_connection_paraphrases_stability():
    """Verify that semantically equivalent paraphrases of Verify Internet Connection
    stably select an appropriate Wi-Fi candidate rather than gesture or Bluetooth/tethering.
    """
    paraphrases = [
        "Navigate to and open Settings. Tap Connections. Tap Wi-Fi.",
        "Open Settings, go to Connections, then Wi-Fi and confirm connectivity.",
        "Use Quick Settings or open the Wi-Fi settings page to verify the network.",
        "Swipe down from the top of the screen to open the Quick settings panel. Touch and hold the Wi-Fi icon to check connection status. Alternatively, open Settings. Tap Connections. Tap Wi-Fi.",
    ]
    for step in paraphrases:
        results = match_deeplink(step, top_k=3, action_name="Verify Internet Connection")
        assert len(results) >= 1
        top = results[0]
        # Must match a genuine Wi-Fi candidate
        assert "wi-fi" in top.get("description", "").lower() or "wifi" in top.get("message", "").lower()
        # Must not be gesture or tethering
        assert top.get("id") not in ("DL-0460", "DL-0100", "DL-0088", "DL-0089")
        assert top.get("score", 0) >= 0.5


def test_clear_app_cache_and_data_returns_no_match():
    """Clear App Cache and Data has no system-level catalog deeplink and must
    cleanly return no match (empty list) rather than selecting spurious settings.
    """
    paraphrases = [
        "Navigate to Settings. Tap Apps. Select your email app. Tap Storage. Tap Clear cache. Tap Clear data, and then tap OK.",
        "Open Settings, go to Apps, find the email app, tap Storage, then tap Clear Cache.",
        "Go to Settings > Apps > Email > Storage and clear app data and cache.",
        "Navigate to and open Settings. Tap Apps. Select your email app. Tap Storage. Tap Clear cache.",
    ]
    for step in paraphrases:
        results = match_deeplink(step, top_k=3, action_name="Clear App Cache and Data")
        assert len(results) == 0


def test_paraphrase_stability_across_varied_wording():
    """Verify that retrieval remains stable across varied procedural wording for the same intent."""
    touch_enable_paraphrases = [
        "Open Settings. Tap Display. Turn on Touch sensitivity.",
        "Go to Settings, select Display, and enable the Touch sensitivity switch.",
        "Navigate to Settings > Display and switch on Touch sensitivity for screen protectors.",
    ]
    for p in touch_enable_paraphrases:
        results = match_deeplink(p, top_k=1, action_name="Enable Touch Sensitivity")
        assert results[0]["id"] == "DL-0126"
        assert results[0]["originalType"] == "onURL"

    touch_disable_paraphrases = [
        "Open Settings. Tap Display. Turn off Touch sensitivity.",
        "Go to Settings, select Display, and disable the Touch sensitivity switch.",
        "Navigate to Settings > Display and switch off Touch sensitivity.",
    ]
    for p in touch_disable_paraphrases:
        results = match_deeplink(p, top_k=1, action_name="Disable Touch Sensitivity")
        assert results[0]["id"] == "DL-0125"
        assert results[0]["originalType"] == "offURL"


def test_aspect_ratio_action_returns_no_match():
    """Aspect ratio / Smart View display ratio actions have no matching catalog
    deeplink and must cleanly return no match (empty list) rather than
    selecting unrelated audio/speaker or layout candidates.
    """
    aspect_ratio_paraphrases = [
        (
            "Adjust Phone Aspect Ratio",
            "Tap the Smart View icon on your phone. Tap More options (the three vertical dots). Tap Settings. Tap Phone aspect ratio. Select your desired aspect ratio.",
        ),
        (
            "Change Phone Aspect Ratio",
            "Tap the Smart View icon on your device. Tap More options (the three vertical dots). Tap Settings. Tap Phone aspect ratio. Select desired aspect ratio.",
        ),
        ("Change display aspect ratio", "Tap Phone aspect ratio."),
        ("Adjust screen aspect ratio", "Tap Phone aspect ratio."),
        ("Modify phone display aspect ratio", "Tap Phone aspect ratio."),
        ("Change the phone's display ratio", "Tap Phone aspect ratio."),
        ("Set the desired screen aspect ratio", "Tap Phone aspect ratio."),
    ]
    for act, steps in aspect_ratio_paraphrases:
        results = match_deeplink(steps, top_k=3, action_name=act)
        assert len(results) == 0


def test_factory_reset_retrieval():
    """Factory reset actions must reliably retrieve the reset settings candidate."""
    results = match_deeplink(
        "Navigate to and open Settings. Tap General management. Tap Reset. Tap Factory data reset.",
        top_k=3,
        action_name="Factory Data Reset",
    )
    assert len(results) >= 1
    top = results[0]
    assert "reset" in top.get("message", "").lower() or "reset" in top.get("description", "").lower()
    assert top.get("score", 0) >= 0.5



