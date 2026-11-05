import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "deeplinks.json"

_FUSION_BM25_WEIGHT = 0.4
_FUSION_EMBED_WEIGHT = 0.6
_SCORE_THRESHOLD = 0.5
_MIN_SEMANTIC_CONFIDENCE = 0.38
_MIN_BM25_SCALE = 8.0

_PROCEDURAL_PHRASES = [
    r"\bswipe\s+down(?:\s+from\s+the\s+top(?:\s+of\s+(?:the|your)\s+screen)?)?\b",
    r"\bswipe\s+up\b",
    r"\btouch\s+and\s+hold\b",
    r"\bpress\s+and\s+hold\b",
    r"\bnavigate\s+to(?:\s+and\s+open)?\b",
    r"\bgo\s+to\b",
    r"\btap\s+to\s+open\b",
    r"\balternatively\b",
    r"\band\s+then\b",
    r"\bwhen\s+it\s+appears\b",
    r"\bcheck\s+(?:your\s+)?connection\s+status\b",
    r"\bconfirm\s+(?:your\s+)?connection(?:\s+status)?\b",
    r"\bmore\s+options(?:\s*\([^)]*\))?\b",
    r"\b(?:the\s+)?three\s+vertical\s+dots\b",
    r"\bon\s+(?:your|the)\s+(?:phone|device|tablet)\b",
]

_PROCEDURAL_WORDS = {
    "tap", "taps", "tapping", "swipe", "swipes", "swiping",
    "open", "opens", "opening", "select", "selects", "selecting",
    "choose", "chooses", "choosing", "press", "presses", "pressing",
    "touch", "touches", "touching", "hold", "holds", "holding",
    "scroll", "scrolls", "scrolling", "navigate", "navigates",
    "adjust", "adjusts", "adjusting", "change", "changes", "changing",
    "configure", "configures", "configuring", "modify", "modifies", "modifying",
    "set", "sets", "setting", "settings",
    "phone", "phones", "device", "devices", "tablet", "tablets",
    "option", "options", "icon", "icons", "menu", "menus", "desired",
    "to", "and", "the", "on", "in", "via", "at", "by", "or", "your",
    "this", "that", "these", "those", "a", "an", "is", "it", "of",
    "from", "for", "with", "into", "again", "then", "next",
    "use", "using", "quick",
}


def clean_procedural_text(text: str) -> str:
    cleaned = text
    for pat in _PROCEDURAL_PHRASES:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    tokens = _tokenize(cleaned)
    filtered = [
        t.strip(".,;:!?()[]\"'")
        for t in tokens
        if t.strip(".,;:!?()[]\"'") and t.strip(".,;:!?()[]\"'") not in _PROCEDURAL_WORDS
    ]
    return " ".join(filtered)


def _safe_text(*fields: str | None) -> str:
    return " ".join(f for f in fields if f).strip()


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_index():
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    entries = catalog["deeplinks"]

    corpus_texts: list[str] = []
    for e in entries:
        corpus_texts.append(
            _safe_text(e.get("description"), e.get("message"), e.get("qna_description"))
        )

    tokenized_corpus = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    embeddings = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    return entries, bm25, model, embeddings


_entries, _bm25, _model, _embeddings = _build_index()


# Generic device / OS settings terminology for filtering standard settings
_GENERIC_SETTINGS_VOCAB = {
    "setting", "settings", "option", "options", "menu", "menus", "page", "pages",
    "display", "screen", "screens", "brightness", "adaptive brightness", "screen timeout",
    "font", "fonts", "sound", "sounds", "volume", "vibration", "vibrations", "ringtone",
    "mute", "notification", "notifications", "alert", "alerts",
    "battery", "power", "power saving", "charging", "fast charging", "wireless charging",
    "storage", "memory", "apps", "app", "application", "applications", "default apps",
    "device care", "optimization", "clean", "cache", "data",
    "connection", "connections", "network", "networks",
    "wifi", "wi-fi", "wi", "fi", "bluetooth", "mobile data", "data saver",
    "airplane", "airplane mode", "flight mode", "hotspot", "mobile hotspot", "tethering",
    "nfc", "contactless", "payments", "contactless payments", "wallet",
    "lock", "lock screen", "wallpaper", "wallpapers", "security", "biometrics",
    "fingerprint", "face", "face unlock", "password", "pin", "pattern",
    "touch", "sensitivity", "touch sensitivity", "touch screen",
    "navigation", "navigation bar", "gestures", "gesture", "button", "buttons", "side button",
    "keyboard", "keyboards", "input", "mouse", "cursor",
    "date", "time", "clock", "date and time", "time zone", "24-hour",
    "language", "languages", "speech", "text-to-speech",
    "accessibility", "emergency", "emergency contacts", "sos",
    "backup", "restore", "sync", "auto sync", "account", "accounts",
    "update", "updates", "software update", "system update",
    "reset", "restart", "reboot", "factory reset", "factory data reset",
    "camera", "flashlight", "torch", "sim", "sim manager", "esim",
    "always on display", "aod", "do not disturb", "dnd",
    "privacy", "permissions", "location", "diagnostics",
    "auto", "automatic", "default", "personal", "system", "general", "advanced",
    "mode", "modes", "level", "status", "shortcut", "shortcuts", "info", "details",
    "view", "open", "enable", "disable", "turn", "switch", "adjust", "check", "set",
    "scanning", "tethering", "usage", "protection", "limit", "warning", "sound mode",
    "hearing", "aids", "media"
}

_GENERIC_TITLES = {
    "device settings", "tv settings", "settings page", "quick settings",
    "home screen", "apps screen", "power menu", "navigation bar",
    "always on display", "do not disturb", "mobile hotspot", "device care",
    "sound mode", "system sound", "lock screen"
}

_TITLE_CASE_PATTERN = re.compile(
    r"^([A-Z][a-zA-Z0-9\+\.]+(?:\s+(?:to\s+|for\s+|of\s+|and\s+)?[A-Z][a-zA-Z0-9\+\.]+)+)"
)


def _extract_specialized_feature(entry: dict) -> Optional[tuple[str, set[str]]]:
    """Identify if a catalog entry represents a specialized/branded feature name.

    Returns (feature_name, distinctive_tokens) if specialized, or None if generic settings.
    """
    desc = entry.get("description", "")

    clean = re.sub(
        r"\s+(?:page\s+)?(?:via|in)\s+(?:device|TV)\s+Settings\s+on\s+the\s+device\.", "", desc
    )
    clean = re.sub(
        r"^(?:Opens(?: the)?|Enables|Disables|Updates(?: the)?|Adjusts?|Runs?)\s+", "", clean
    )
    clean = re.sub(r"\s+settings\s+page$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+settings$", "", clean, flags=re.IGNORECASE).strip()

    m_feat = _TITLE_CASE_PATTERN.match(clean)
    if not m_feat:
        return None

    cand = m_feat.group(1).strip()
    cand_lower = cand.lower()
    if cand_lower in _GENERIC_TITLES or cand_lower in _GENERIC_SETTINGS_VOCAB:
        return None

    tokens = [
        t
        for t in re.findall(r"[a-z0-9\+]+", cand_lower)
        if t not in {"to", "for", "of", "and", "in", "on", "the", "a"}
    ]
    distinctive = [t for t in tokens if t not in _GENERIC_SETTINGS_VOCAB]
    if distinctive:
        return cand, set(distinctive)

    return None


_specialized_features: dict[int, tuple[str, set[str]]] = {}
for idx, _entry in enumerate(_entries):
    _sf = _extract_specialized_feature(_entry)
    if _sf:
        _specialized_features[idx] = _sf


_POS_PATTERNS = [
    r"\benables?\b", r"\benabling\b",
    r"\bturn(?:s|ed|ing)?\s+on\b",
    r"\bactivates?\b", r"\bactivating\b",
    r"\bswitch(?:es|ed|ing)?\s+on\b",
]

_NEG_PATTERNS = [
    r"\bdisables?\b", r"\bdisabling\b",
    r"\bturn(?:s|ed|ing)?\s+off\b",
    r"\bdeactivates?\b", r"\bdeactivating\b",
    r"\bswitch(?:es|ed|ing)?\s+off\b",
]


def _detect_polarity(action_name: Optional[str], step_text: str) -> Optional[str]:
    """Detect explicit toggle polarity ('on' or 'off') from actionName and step text.

    Returns None if no explicit polarity signal is found or if ambiguous without
    a clear action intent.
    """
    act_text = (action_name or "").lower()
    st_text = (step_text or "").lower()

    pos_act = any(re.search(p, act_text) for p in _POS_PATTERNS)
    neg_act = any(re.search(p, act_text) for p in _NEG_PATTERNS)
    pos_step = any(re.search(p, st_text) for p in _POS_PATTERNS)
    neg_step = any(re.search(p, st_text) for p in _NEG_PATTERNS)

    if pos_act and not neg_act:
        return "on"
    if neg_act and not pos_act:
        return "off"
    if pos_step and not neg_step:
        return "on"
    if neg_step and not pos_step:
        return "off"
    return None


def match_deeplink(
    step_text: str,
    top_k: int = 3,
    action_name: Optional[str] = None,
) -> List[dict]:
    """Match step text and optional actionName against the deeplink catalog.

    Uses a dual-signal retrieval architecture:
    - actionName acts as a stable, high-level intent anchor.
    - step_text is filtered of procedural noise (tap, swipe, navigate, etc.)
      to extract content-bearing setting and navigation path terms.
    - Dual-signal query combines actionName and content terms for hybrid
      BM25 + dense embedding retrieval.
    - Enforces toggle polarity (on/off) compatibility.
    - Applies a specialized-feature false-positive guard (Phase 5.5 Fix 1).
    - Prefers toggle over sub-screen for explicit Enable/Disable actions (Phase 5.5 Fix 2).
    - Applies a confidence gate (fused >= 0.50 and dense cosine >= 0.38)
      to cleanly return an empty list when no catalog candidate
      is semantically relevant (e.g. Clear App Cache and Data).
    """
    polarity = _detect_polarity(action_name, step_text)

    cleaned_steps = clean_procedural_text(step_text)

    if action_name:
        if cleaned_steps:
            full_query = f"{action_name}. {cleaned_steps}"
        else:
            full_query = action_name

        query_tokens = [
            t.strip(".,;:!?()[]\"'")
            for t in _tokenize(full_query)
            if t.strip(".,;:!?()[]\"'") and t.strip(".,;:!?()[]\"'") not in _PROCEDURAL_WORDS
        ]
        if not query_tokens:
            query_tokens = _tokenize(step_text)
    else:
        full_query = step_text
        query_tokens = [
            t.strip(".,;:!?()[]\"'")
            for t in _tokenize(cleaned_steps)
            if t.strip(".,;:!?()[]\"'") and t.strip(".,;:!?()[]\"'") not in _PROCEDURAL_WORDS
        ]
        if not query_tokens:
            query_tokens = _tokenize(step_text)

    bm25_raw = np.array(_bm25.get_scores(query_tokens), dtype=np.float32)
    bm25_max = bm25_raw.max()
    bm25_denom = max(bm25_max, _MIN_BM25_SCALE)
    bm25_norm = bm25_raw / bm25_denom if bm25_denom > 0 else bm25_raw

    query_emb = _model.encode([full_query], normalize_embeddings=True, show_progress_bar=False)
    cosine_sim = np.dot(_embeddings, query_emb[0])

    fused = _FUSION_BM25_WEIGHT * bm25_norm + _FUSION_EMBED_WEIGHT * cosine_sim

    # Adjust scores based on polarity
    adjusted_scores = fused.copy()
    if polarity is not None:
        for idx, e in enumerate(_entries):
            otype = e.get("originalType")
            desc = e.get("description", "").lower()
            msg = e.get("message", "").lower()

            cand_dir = None
            if otype == "onURL" or desc.startswith("enables ") or msg.startswith("enable "):
                cand_dir = "on"
            elif otype == "offURL" or desc.startswith("disables ") or msg.startswith("disable "):
                cand_dir = "off"

            if polarity == "on":
                if cand_dir == "off":
                    adjusted_scores[idx] = -1.0
                elif cand_dir == "on":
                    adjusted_scores[idx] += 0.05
            elif polarity == "off":
                if cand_dir == "on":
                    adjusted_scores[idx] = -1.0
                elif cand_dir == "off":
                    adjusted_scores[idx] += 0.05

    ranked_indices = np.argsort(adjusted_scores)[::-1]

    # Query text for specialized entity presence check (Fix 1)
    query_text_lower = f"{action_name or ''} {step_text}".lower()

    valid_candidates = []
    for idx in ranked_indices:
        score = adjusted_scores[idx]
        if score < _SCORE_THRESHOLD or cosine_sim[idx] < _MIN_SEMANTIC_CONFIDENCE:
            continue

        e = _entries[idx]

        # Fix 1: Specialized-feature false-positive guard.
        # If candidate represents a specialized/branded feature (e.g. Music Share), require that
        # the feature's own name (or distinctive token) actually appears in the action name or step text.
        if idx in _specialized_features:
            feat_name, feat_tokens = _specialized_features[idx]
            name_present = (feat_name.lower() in query_text_lower) or any(
                tok in query_text_lower for tok in feat_tokens
            )
            if not name_present:
                continue

        valid_candidates.append({
            "id": e["id"],
            "deeplink": e["deeplink"],
            "description": e.get("description", ""),
            "message": e.get("message", ""),
            "originalType": e.get("originalType", ""),
            "score": float(score),
            "validation": e.get("validation"),
        })
        if len(valid_candidates) >= max(top_k, 5):
            break

    if not valid_candidates:
        return []

    # Fix 2: Prefer toggle over sub-screen for explicit Enable/Disable actions
    if action_name and valid_candidates:
        act_lower = action_name.lower().strip()
        is_enable = bool(re.match(r"^(?:enable|turn\s+on|activate|switch\s+on)\b", act_lower))
        is_disable = bool(re.match(r"^(?:disable|turn\s+off|deactivate|switch\s+off)\b", act_lower))

        if (is_enable or is_disable) and valid_candidates[0].get("originalType") == "onClickURL":
            target_type = "onURL" if is_enable else "offURL"
            topic_terms = [
                t
                for t in re.findall(r"[a-z0-9]+", act_lower)
                if t
                not in {
                    "enable",
                    "disable",
                    "turn",
                    "on",
                    "off",
                    "activate",
                    "deactivate",
                    "switch",
                    "settings",
                    "setting",
                    "mode",
                }
            ]
            if topic_terms:
                for c_idx, cand in enumerate(valid_candidates[1:], start=1):
                    if cand.get("originalType") == target_type:
                        cand_text = f"{cand.get('description', '')} {cand.get('message', '')}".lower()
                        if any(term in cand_text for term in topic_terms):
                            preferred = valid_candidates.pop(c_idx)
                            valid_candidates.insert(0, preferred)
                            break

    return valid_candidates[:top_k]



