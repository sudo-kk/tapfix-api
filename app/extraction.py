import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.deeplink_match import (
    _NEG_PATTERNS,
    _POS_PATTERNS,
    _detect_polarity,
    match_deeplink,
)
from app.schema import (
    Action,
    Condition,
    ContextDeeplinkResponse,
    Deeplink,
    Goal,
    ResultTypes,
    StepGroup,
    ValidationDeepLink,
    actionCategory,
)
from app.validators import (
    scrub_urls,
    validate_full_response,
)

logger = logging.getLogger(__name__)

load_dotenv()

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        _client = genai.Client(api_key=api_key)
    return _client


_SYSTEM_PROMPT = """You are a Samsung mobile device troubleshooting assistant.
Given a user query, a SIIS (Samsung internal knowledge store) document title, and SIIS document content, extract a structured troubleshooting plan.

CRITICAL RULES:
1. STRICT GROUNDING: Derive the troubleshooting plan ONLY from the provided SIIS content.
   - Do NOT invent troubleshooting steps, actions, settings, or procedures that are not supported by the SIIS content.
   - Do NOT introduce any external facts, unsupported settings, or ungrounded suggestions.
   - When assessing whether the SIIS content contains relevant troubleshooting steps:
     * Harmless wording differences between the reported symptom and the SIIS title (e.g. a display that flashes, glitches, or blanks when connecting a charger vs. a SIIS document titled "Blank or black display on a Samsung phone or tablet" that provides steps to inspect charger, cable, and port for damage or corrosion, check LDI, force restart, charge, test power-on, and contact support) are relevant. Extract only those directly grounded steps without inventing any specific diagnosis for the flashing.
     * In contrast, genuine topical mismatches where the SIIS document covers completely unrelated features (e.g. Multi window or App pairs for a dark screen or floating menu, Camera shutter speed for a broken folding hinge, or Smart Switch data transfer for a blank tablet screen) must return: {"contexts": []}.
   - If the SIIS content does not contain actionable troubleshooting steps relevant to the user query, return: {"contexts": []}.

2. ZERO URL LEAKS: Do NOT include any URLs, web links, or domain names (e.g. samsung.com, http, https, www) anywhere in your output.

3. ACTION CATEGORIZATION RULES (STRICT):
   - "manual": MUST be used for ALL physical or hardware actions, including:
     * Pressing or holding physical buttons (e.g. Power, Side, or Volume buttons)
     * Force restart / hard reboot via hardware key combinations (e.g. Power + Volume Down)
     * Powering a device on or off using physical buttons
     * Connecting or disconnecting a charger, USB cable, mouse, keyboard, or adapter
     * Waiting for the device to charge (e.g. charging for 1 hour)
     * Inserting or removing physical hardware (SIM tray, SD card, battery)
     * Physical inspection (e.g. checking for cracks, inspecting Liquid Damage Indicator)
     * Rebooting into Safe Mode or restarting via the power menu
     * Contacting Samsung Support, visiting a service center, or scheduling repairs
     * Any action that fundamentally requires physical interaction and CANNOT be opened by a Samsung Settings deeplink.
     IMPORTANT: Do NOT classify an action as "auto" merely because it contains taps, navigation, or multiple instructions. If it requires physical hardware manipulation, it MUST be "manual".
   - "auto": Use ONLY for genuine software/Settings actions performed on-screen inside device Settings or built-in system apps (e.g. Wi-Fi settings, Display settings, Sound settings, Apps settings, Cloud backup settings).
   - "critical": Use ONLY for genuinely critical software/settings actions (e.g. Factory Data Reset). Do NOT use for physical actions.

4. SCHEMA CONSTRAINTS:
   Return a JSON object with a "contexts" array containing 0 or 1 Goal object:
   - "goal": MUST match the exact pattern: "Follow these steps to perform this <Topic> Troubleshooting" or "Follow these steps to perform this <Topic> Configuration".
     Example: "Follow these steps to perform this Email Connection Troubleshooting"
   - "title": MUST be EXACTLY 2 or 3 words describing the issue or topic.
     Example: "Email connection issue" or "Blank display"
   - "score": A float between 0.0 and 1.0 indicating relevance confidence, e.g. 0.95.
   - "actions": A list of Action objects. Each Action represents one logical task or physical screen:
     - "actionName": Concise name for the action (e.g. "Check Internet Connection", "Clear App Cache", "Restart In Safe Mode", "Visit Service Center").
     - "description": MUST be EXACTLY 5, 6, or 7 words, AND MUST start with "It will".
       Count words carefully. Examples of valid 5-7 word descriptions:
       - "It will verify active internet connection" (6 words)
       - "It will clear application cache data" (6 words)
       - "It will restart device in safe mode" (6 words)
       - "It will schedule Samsung repair service" (6 words)
       - "It will inspect device for liquid damage" (6 words)
       - "It will force restart your mobile device" (6 words)
     - "category": MUST be "manual", "auto", or "critical" following the categorization rules above.
     - "stepGroups": A list of StepGroup objects. Each has:
       - "steps": A list of imperative step strings, one action/tap per line.
         Example: ["Navigate to and open Settings.", "Tap Apps.", "Select your email app.", "Tap Storage.", "Tap Clear cache."]
         NEVER include URLs or website links in any step.

5. NO CONTRADICTORY TOGGLE ACTIONS:
   - Do not generate two separate actions that toggle the same underlying setting in opposite directions (e.g. both 'Enable X' and 'Disable X') within a single troubleshooting plan. If the SIIS content suggests trying a setting in one direction as a fix, only include that one direction. If the SIIS content genuinely indicates the user should try toggling a setting and then toggling it back to test, express this as a single action with sequential steps, not two separate actions.
"""


def _build_user_prompt(query: str, siis_title: str, siis_content: str) -> str:
    return (
        f"User Query:\n{query}\n\n"
        f"SIIS Document Title:\n{siis_title}\n\n"
        f"SIIS Document Content:\n{siis_content}\n\n"
        "Extract the structured troubleshooting plan in JSON conforming strictly to the rules."
    )


def _parse_llm_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    result = json.loads(text.strip())
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        raise ValueError(
            f"Malformed LLM JSON: expected dict or single-dict list, got list of length {len(result)}"
        )
    if not isinstance(result, dict):
        raise ValueError(
            f"Malformed LLM JSON: expected dict, got {type(result).__name__}"
        )
    return result



# Semantic keyword pattern covering concepts like restart, reboot, power-cycle, safe mode, force restart
_DISRUPTIVE_PATTERN = re.compile(
    r"\b(restart\w*|reboot\w*|power[\s-]*cycl\w*|safe[\s-]*mode)\b",
    re.IGNORECASE,
)


def is_disruptive_action(action: Action) -> bool:
    """Detect whether a manual action is disruptive (e.g. restart, reboot, power cycle, safe mode)."""
    text = f"{action.actionName} {action.description}"
    return bool(_DISRUPTIVE_PATTERN.search(text))


def _action_sort_key(action: Action) -> int:
    """Return priority for deterministic action sequencing:
    0: auto
    1: ordinary manual
    2: disruptive manual (e.g. restart, reboot, safe mode)
    3: critical (e.g. factory data reset)
    """
    cat = action.category
    if cat == actionCategory.auto:
        return 0
    elif cat == actionCategory.manual:
        if is_disruptive_action(action):
            return 2
        return 1
    elif cat == actionCategory.critical:
        return 3
    return 1


_TOGGLE_VERB_PREFIX = re.compile(
    r"^(?:enable|enabling|disable|disabling|turn\s+(?:[a-z]+\s+)?on|turn\s+(?:[a-z]+\s+)?off|turn|activate|activating|deactivate|deactivating|switch\s+(?:[a-z]+\s+)?on|switch\s+(?:[a-z]+\s+)?off|switch|toggle)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_TOGGLE_SUFFIX = re.compile(
    r"\s+(?:on|off|enabled|disabled|setting|settings|feature|option|mode)$",
    re.IGNORECASE,
)


def _normalize_setting_name(action_name: str) -> str:
    """Extract a normalized setting name from an action name by stripping toggle verbs, articles, and suffixes."""
    text = action_name.strip()
    text = _TOGGLE_VERB_PREFIX.sub("", text)
    text = _TOGGLE_SUFFIX.sub("", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(".:,;!- ")
    return re.sub(r"\s+", " ", text).strip().lower()


def _get_action_polarity(action: Action) -> Optional[str]:
    """Detect whether an action has an explicit toggle polarity ('on' vs 'off')."""
    step_text = " ".join(" ".join(sg.steps) for sg in action.stepGroups)
    full_step_and_desc = f"{action.description} {step_text}"
    return _detect_polarity(action.actionName, full_step_and_desc)


def _prefer_polarity_from_siis(setting: str, siis_content: str) -> Optional[str]:
    """Check if SIIS content determinably indicates a toggle direction ('on' vs 'off') for a setting.

    Returns 'on', 'off', or None if ambiguous, not found, or mentions both.
    """
    if not siis_content or not setting:
        return None

    matches = list(re.finditer(re.escape(setting), siis_content, re.IGNORECASE))
    if not matches:
        return None

    has_pos = False
    has_neg = False

    for m in matches:
        start = max(0, m.start() - 150)
        end = min(len(siis_content), m.end() + 150)
        window = siis_content[start:end].lower()

        if any(re.search(p, window) for p in _POS_PATTERNS):
            has_pos = True
        if any(re.search(p, window) for p in _NEG_PATTERNS):
            has_neg = True

    if has_pos and not has_neg:
        return "on"
    if has_neg and not has_pos:
        return "off"
    return None


def _filter_contradictory_toggles(actions: List[Action], siis_content: str = "") -> List[Action]:
    """Filter out contradictory toggle actions on the same setting within a troubleshooting plan.

    When two actions toggle the same underlying setting in opposite directions:
    - Keep only ONE of the two actions.
    - Prefer the direction actually indicated by SIIS content (if determinable).
    - Otherwise prefer whichever the LLM generated first.
    - Drop the redundant/contradictory opposite-direction action entirely.
    - Log diagnostic message when guard fires.
    """
    kept_actions: List[Action] = []
    # Map normalized_setting -> (index_in_kept_actions, action, polarity)
    seen_toggles: Dict[str, Tuple[int, Action, str]] = {}

    for action in actions:
        polarity = _get_action_polarity(action)
        if polarity not in ("on", "off"):
            kept_actions.append(action)
            continue

        setting = _normalize_setting_name(action.actionName)
        if not setting:
            kept_actions.append(action)
            continue

        if setting not in seen_toggles:
            seen_toggles[setting] = (len(kept_actions), action, polarity)
            kept_actions.append(action)
        else:
            prev_idx, prev_action, prev_polarity = seen_toggles[setting]
            if polarity == prev_polarity:
                # Duplicate action in same direction
                logger.info(
                    f"Dropping redundant duplicate toggle action '{action.actionName}' for setting '{setting}' "
                    f"(already retained '{prev_action.actionName}')"
                )
            else:
                # Contradictory opposite-direction pair detected!
                preferred = _prefer_polarity_from_siis(setting, siis_content)
                if preferred == polarity and preferred != prev_polarity:
                    # Current action matches preferred direction from SIIS, replace previous
                    logger.info(
                        f"Contradictory toggle pair detected for setting '{setting}': replacing '{prev_action.actionName}' "
                        f"({prev_polarity}) with '{action.actionName}' ({polarity}) per SIIS preference"
                    )
                    kept_actions[prev_idx] = action
                    seen_toggles[setting] = (prev_idx, action, polarity)
                else:
                    # Previous action kept (either SIIS preferred prev_polarity, or SIIS ambiguous -> keep first generated)
                    logger.info(
                        f"Contradictory toggle pair detected for setting '{setting}': dropping '{action.actionName}' "
                        f"({polarity}) in favor of earlier '{prev_action.actionName}' ({prev_polarity})"
                    )

    return kept_actions


def _build_response_tree(parsed: dict, siis_content: str = "") -> ContextDeeplinkResponse:
    """Build and enrich the Pydantic response tree from LLM JSON output.

    Enforces:
    - URL scrubbing on all steps.
    - Deeplink matching with dual-signal query & polarity filtering.
    - Multi-StepGroup safety: software actions lacking confident matches for all step groups become manual with null deeplinks.
    - Contradictory toggle deduplication (Phase 5.3a): prevents opposite toggle directions for same setting.
    - Deterministic action sequencing (auto -> ordinary manual -> disruptive manual -> critical).
    """
    raw_contexts = parsed.get("contexts", [])
    goals: List[Goal] = []

    for c in raw_contexts:
        goal_text = c.get("goal", "")
        title_text = c.get("title", "")
        score_val = float(c.get("score", 0.0))

        actions: List[Action] = []
        for a in c.get("actions", []):
            action_name = a.get("actionName", "")
            action_desc = a.get("description", "")
            cat_raw = str(a.get("category", "manual")).lower()

            if cat_raw == "auto":
                cat = actionCategory.auto
            elif cat_raw == "critical":
                cat = actionCategory.critical
            else:
                cat = actionCategory.manual

            raw_sgs = a.get("stepGroups", [])
            processed_sgs = []
            all_sgs_confident = True if (cat != actionCategory.manual and raw_sgs) else False

            for sg in raw_sgs:
                raw_steps = sg.get("steps", [])
                clean_steps = [scrub_urls(s) for s in raw_steps if scrub_urls(s)]
                if not clean_steps:
                    continue

                full_step_text = " ".join(clean_steps)

                if cat == actionCategory.manual:
                    actionable_dl = None
                    val_dl = None
                    all_sgs_confident = False
                else:
                    matches = match_deeplink(full_step_text, top_k=1, action_name=action_name)
                    if not matches or not matches[0].get("deeplink") or matches[0].get("deeplink") == "bixby://dummy_positive":
                        actionable_dl = None
                        val_dl = None
                        all_sgs_confident = False
                    else:
                        top_match = matches[0]
                        actionable_dl = Deeplink(
                            deeplink=top_match["deeplink"],
                            description=top_match.get("description") or "Open the relevant device settings screen",
                            message=top_match.get("message") or "",
                            originalType=top_match.get("originalType") or None,
                        )

                        val_info = top_match.get("validation")
                        if isinstance(val_info, dict) and val_info.get("deeplink") and val_info.get("key"):
                            orig_type = top_match.get("originalType")
                            if orig_type == "onURL":
                                r_type = ResultTypes.boolean
                                cond = Condition.equal
                                val = "True"
                            elif orig_type == "offURL":
                                r_type = ResultTypes.boolean
                                cond = Condition.equal
                                val = "False"
                            else:
                                r_type = None
                                cond = None
                                val = None

                            val_dl = ValidationDeepLink(
                                deeplink=val_info["deeplink"],
                                key=val_info["key"],
                                resultType=r_type,
                                condition=cond,
                                value=val,
                            )
                        else:
                            val_dl = None

                processed_sgs.append({
                    "clean_steps": clean_steps,
                    "actionable_dl": actionable_dl,
                    "val_dl": val_dl,
                })

            if not processed_sgs:
                continue

            # Multi-StepGroup categorization safety:
            # If any StepGroup lacks a confident match, the action cannot remain auto/critical.
            # It must be manual, and per Phase 5.1 all StepGroups must have null deeplinks.
            if cat == actionCategory.manual or not all_sgs_confident:
                final_cat = actionCategory.manual
                step_groups = [
                    StepGroup(
                        steps=item["clean_steps"],
                        actionableDeeplink=None,
                        validationDeeplink=None,
                    )
                    for item in processed_sgs
                ]
            else:
                final_cat = cat
                step_groups = [
                    StepGroup(
                        steps=item["clean_steps"],
                        actionableDeeplink=item["actionable_dl"],
                        validationDeeplink=item["val_dl"],
                    )
                    for item in processed_sgs
                ]

            actions.append(
                Action(
                    actionName=action_name,
                    description=action_desc,
                    stepGroups=step_groups,
                    category=final_cat,
                )
            )

        # Phase 5.3a: Guard against contradictory toggle-direction actions
        actions = _filter_contradictory_toggles(actions, siis_content=siis_content)

        # Phase 5.2 Deterministic Action Sequencing:
        # 0: auto
        # 1: ordinary manual
        # 2: disruptive manual (e.g. restart, reboot, safe mode)
        # 3: critical (e.g. factory data reset)
        # Python's sort is stable: original order is preserved within the same priority.
        actions.sort(key=_action_sort_key)

        goals.append(
            Goal(
                goal=goal_text,
                title=title_text,
                score=score_val,
                actions=actions,
            )
        )

    return ContextDeeplinkResponse(contexts=goals)


def extract_goal(query: str, siis_title: str, siis_content: str) -> dict:
    """Extract a validated troubleshooting plan from SIIS content using Gemini.

    Retries once on validation failure with error feedback.
    Catches errors per query so failures do not abort batch execution.
    """
    try:
        client = _get_client()
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return {
            "contexts": [],
            "fallback": "no_match",
            "_meta": {
                "first_attempt_valid": False,
                "retry_performed": False,
                "final_valid": True,
                "error": str(e),
            },
        }

    user_prompt = _build_user_prompt(query, siis_title, siis_content)

    first_attempt_valid = False
    retry_performed = False
    failures: List[str] = []

    try:
        # Attempt 1
        resp_1 = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        try:
            parsed_1 = _parse_llm_json(resp_1.text or "{}")
            response_tree_1 = _build_response_tree(parsed_1, siis_content=siis_content)
            is_valid_1, failures_1 = validate_full_response(response_tree_1)
        except Exception as parse_err:
            is_valid_1 = False
            failures_1 = [f"JSON parsing/structure error: {parse_err}"]

        if is_valid_1:
            first_attempt_valid = True
            result = response_tree_1.model_dump()
            result["_meta"] = {
                "first_attempt_valid": True,
                "retry_performed": False,
                "final_valid": True,
                "failures": [],
            }
            return result

        # Attempt 1 failed validation -> Retry exactly once
        retry_performed = True
        failures = failures_1
        retry_prompt = (
            f"{user_prompt}\n\n"
            f"IMPORTANT: The previous output failed validation with the following errors:\n"
            + "\n".join(f"- {f}" for f in failures_1)
            + "\nPlease regenerate the complete JSON fixing every error. "
            "Ensure 'goal' matches the required format, 'title' is exactly 2-3 words, "
            "and every 'description' is exactly 5-7 words starting with 'It will'."
        )

        resp_2 = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        try:
            parsed_2 = _parse_llm_json(resp_2.text or "{}")
            response_tree_2 = _build_response_tree(parsed_2, siis_content=siis_content)
            is_valid_2, failures_2 = validate_full_response(response_tree_2)
        except Exception as parse_err:
            is_valid_2 = False
            failures_2 = [f"JSON parsing/structure error on retry: {parse_err}"]

        if is_valid_2:
            result = response_tree_2.model_dump()
            result["_meta"] = {
                "first_attempt_valid": False,
                "retry_performed": True,
                "final_valid": True,
                "failures": failures_2,
            }
            return result

        # Failed after retry: do not return unvalidated output
        logger.warning(f"Validation failed after retry for query: {query[:50]}... Failures: {failures_2}")
        empty_resp = ContextDeeplinkResponse(contexts=[])
        result = empty_resp.model_dump()
        result["fallback"] = "validation_failed"
        result["_meta"] = {
            "first_attempt_valid": False,
            "retry_performed": True,
            "final_valid": False,
            "failures": failures_2,
        }
        return result

    except Exception as e:
        logger.error(f"Error processing query '{query[:50]}...': {e}")
        empty_resp = ContextDeeplinkResponse(contexts=[])
        result = empty_resp.model_dump()
        result["fallback"] = "error"
        result["_meta"] = {
            "first_attempt_valid": first_attempt_valid,
            "retry_performed": retry_performed,
            "final_valid": False,
            "error": str(e),
            "failures": failures,
        }
        return result
