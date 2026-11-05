import pytest
from app.deeplink_match import match_deeplink


def test_fix1_disconnect_bluetooth_audio_rejects_music_share():
    """Requirement 1: Exact 'Disconnect Bluetooth Audio' step text/action name from
    unseen_2_audio must NOT match Music Share; it should match a generic Bluetooth-related
    deeplink if one exists, or fall back to manual/null.
    """
    action_name = "Disconnect Bluetooth Audio"
    step_text = (
        "Swipe down from the top of the screen to open the Quick settings panel. "
        "Turn off Bluetooth or disconnect connected audio devices."
    )

    matches = match_deeplink(step_text, top_k=3, action_name=action_name)
    assert matches, "Expected at least one matching candidate for Bluetooth action"

    top = matches[0]
    # Verify that Music Share (DL-0127) is NOT matched
    assert top["id"] != "DL-0127", f"Expected Music Share to be excluded, but matched {top['id']}"
    assert "music share" not in top["description"].lower(), f"Matched specialized feature: {top['description']}"

    # Verify that generic Bluetooth off toggle (DL-0494) is selected
    assert top["id"] == "DL-0494", f"Expected DL-0494 (generic Bluetooth off), got {top['id']}"
    assert top["originalType"] == "offURL"
    assert "bluetooth" in top["description"].lower()


def test_fix1_music_share_accepted_when_explicitly_requested():
    """Verify that Music Share is still matched when the query explicitly names it."""
    action_name = "Disable Music Share"
    step_text = "Navigate to Settings. Tap Connections. Turn off Music Share."

    matches = match_deeplink(step_text, top_k=1, action_name=action_name)
    assert matches, "Expected match for explicit Music Share request"
    assert matches[0]["id"] == "DL-0127"
    assert matches[0]["originalType"] == "offURL"


def test_fix2_enable_nfc_prefers_toggle_over_subscreen():
    """Requirement 2: Exact 'Enable NFC Settings' step text/action name from
    unseen_3_nfc should prefer the top-level NFC on/off toggle (DL-0177)
    over Default Wallet App sub-screen (DL-0175).
    """
    action_name = "Enable NFC Settings"
    step_text = (
        "Navigate to and open Settings. Tap Connections. "
        "Tap the switch next to NFC and contactless payments to turn it on. "
        "Tap Contactless payments. Select your default payment app."
    )

    matches = match_deeplink(step_text, top_k=3, action_name=action_name)
    assert matches, "Expected matching candidates for NFC action"

    top = matches[0]
    # Must prefer top-level toggle DL-0177 (onURL) over DL-0175 (onClickURL sub-screen)
    assert top["id"] == "DL-0177", f"Expected DL-0177 (onURL NFC switch), but got {top['id']}"
    assert top["originalType"] == "onURL"
    assert "nfc" in top["description"].lower()

    # Confirm that DL-0175 is still present among candidates as the next option
    candidate_ids = [m["id"] for m in matches]
    assert "DL-0175" in candidate_ids, "Expected DL-0175 to remain in top candidates"


def test_existing_passing_cases_unchanged():
    """Requirement 3: Confirm all existing passing cases are UNCHANGED:
    - Enable Bluetooth
    - Disable Bluetooth
    - Enable Touch Sensitivity
    - Disable Touch Sensitivity
    - Factory Data Reset
    - Configure Dolby Atmos Settings (specialized feature present in query)
    - Unmatched action (Clear App Cache and Data) falls back to empty list
    """
    # 1. Enable Bluetooth
    m_bt_on = match_deeplink(
        "Open Settings, tap Connections, and toggle Bluetooth to on.",
        top_k=1,
        action_name="Enable Bluetooth"
    )
    assert m_bt_on and m_bt_on[0]["id"] == "DL-0495"
    assert m_bt_on[0]["originalType"] == "onURL"

    # 2. Disable Bluetooth
    m_bt_off = match_deeplink(
        "Open Settings, tap Connections, and toggle Bluetooth to off.",
        top_k=1,
        action_name="Disable Bluetooth"
    )
    assert m_bt_off and m_bt_off[0]["id"] == "DL-0494"
    assert m_bt_off[0]["originalType"] == "offURL"

    # 3. Enable Touch Sensitivity
    m_ts_on = match_deeplink(
        "Navigate to and open Settings. Tap Display. Tap the switch next to Touch sensitivity to turn it on.",
        top_k=1,
        action_name="Enable Touch Sensitivity"
    )
    assert m_ts_on and m_ts_on[0]["id"] == "DL-0126"
    assert m_ts_on[0]["originalType"] == "onURL"

    # 4. Disable Touch Sensitivity
    m_ts_off = match_deeplink(
        "Navigate to and open Settings. Tap Display. Tap the switch next to Touch sensitivity to turn it off.",
        top_k=1,
        action_name="Disable Touch Sensitivity"
    )
    assert m_ts_off and m_ts_off[0]["id"] == "DL-0125"
    assert m_ts_off[0]["originalType"] == "offURL"

    # 5. Factory Data Reset
    m_fdr = match_deeplink(
        "Navigate to Settings. Tap General management. Tap Reset. Tap Auto factory reset.",
        top_k=1,
        action_name="Factory Data Reset"
    )
    assert m_fdr and m_fdr[0]["id"] == "DL-0022"

    # 6. Configure Dolby Atmos Settings (specialized feature name in query)
    m_atmos = match_deeplink(
        "Navigate to Settings, tap Sounds and vibration, select Sound quality and effects, and tap the switch next to Dolby Atmos to turn it off.",
        top_k=1,
        action_name="Configure Dolby Atmos Settings"
    )
    assert m_atmos and m_atmos[0]["id"] == "DL-0464"
    assert "dolby atmos" in m_atmos[0]["description"].lower()

    # 7. Unmatched action: Clear App Cache and Data
    m_cache = match_deeplink(
        "Navigate to Settings. Tap Apps. Tap Clear cache. Tap Clear data.",
        top_k=1,
        action_name="Clear App Cache and Data"
    )
    assert m_cache == [], "Expected empty list for unmatched feature without catalog match"
