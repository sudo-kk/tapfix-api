import re
from typing import List, Tuple

from app.schema import ContextDeeplinkResponse


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_FULL_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_WWW_RE = re.compile(r"www\.[^\s)\]>\"']+")
_BARE_DOMAIN_COM = re.compile(r"\b\S+\.com\b")
_BARE_DOMAIN_HTML = re.compile(r"\b\S+\.html\b")


def scrub_urls(text: str) -> str:
    """Strip every trace of URLs, markdown links, and domain-like strings."""
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub("", text)
    text = _FULL_URL_RE.sub("", text)
    text = _WWW_RE.sub("", text)
    text = _BARE_DOMAIN_COM.sub("", text)
    text = _BARE_DOMAIN_HTML.sub("", text)

    text = re.sub(r"  +", " ", text).strip()
    return text



_GOAL_RE = re.compile(
    r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)\.?$"
)


def validate_goal_format(goal: str) -> bool:
    """Goal string must match the canonical sentence pattern."""
    return bool(_GOAL_RE.match(goal))


def validate_title(title: str) -> bool:
    """Title must be exactly 2–3 words."""
    word_count = len(title.split())
    return 2 <= word_count <= 3


def validate_description(desc: str) -> bool:
    """Description must be exactly 5–7 words AND start with 'It will'."""
    if not desc.startswith("It will"):
        return False
    word_count = len(desc.split())
    return 5 <= word_count <= 7


def validate_score(score: float) -> bool:
    """Score must be in [0.0, 1.0]."""
    return 0.0 <= score <= 1.0


def validate_full_response(
    resp: ContextDeeplinkResponse,
) -> Tuple[bool, List[str]]:
    """Run every validator across the full response tree.

    Returns (is_valid, list_of_failure_messages).
    """
    failures: List[str] = []

    for g_idx, goal in enumerate(resp.contexts):
        if not validate_goal_format(goal.goal):
            failures.append(
                f"contexts[{g_idx}].goal: does not match required format"
            )

        if not validate_title(goal.title):
            failures.append(
                f"contexts[{g_idx}].title: must be 2–3 words, got '{goal.title}'"
            )

        if not validate_score(goal.score):
            failures.append(
                f"contexts[{g_idx}].score: must be 0.0–1.0, got {goal.score}"
            )

        for a_idx, action in enumerate(goal.actions):
            prefix = f"contexts[{g_idx}].actions[{a_idx}]"

            if not validate_description(action.description):
                failures.append(
                    f"{prefix}.description: must be 5–7 words starting with "
                    f"'It will', got '{action.description}'"
                )


            for sg_idx, sg in enumerate(action.stepGroups):
                for s_idx, step in enumerate(sg.steps):
                    scrubbed = scrub_urls(step)
                    if scrubbed != step:
                        failures.append(
                            f"{prefix}.stepGroups[{sg_idx}].steps[{s_idx}]: "
                            f"contains URL/link content"
                        )

    return (len(failures) == 0, failures)
