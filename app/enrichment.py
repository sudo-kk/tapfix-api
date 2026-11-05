import json
import logging
import os
import re
from typing import List, Optional, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

load_dotenv()

_client: Optional[genai.Client] = None


def _get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _normalize_for_comparison(text: str) -> str:
    """Normalize query text for duplicate detection.

    - Strips leading numbered list prefixes (e.g. '1. ', '2. ')
    - Strips surrounding quotes
    - Collapses multiple whitespaces into a single space
    - Converts to lowercase
    - Strips trailing punctuation (. ? ! , : ;)
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = cleaned.strip('"\'')
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower().rstrip(".?!,:; ")


def _clean_source_text(query: str) -> str:
    """Strip leading numbering and quotes from source query while preserving case."""
    cleaned = query.strip()
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = cleaned.strip('"\'')
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _generate_deterministic_fallbacks(query: str) -> List[str]:
    """Generate high-quality, semantically faithful fallback variations deterministically.

    Preserves exact device context, model names, and symptoms without inventing technical facts.
    """
    cleaned = _clean_source_text(query)
    if not cleaned:
        return []

    fallbacks: List[str] = []

    # Check for question starting with How do I / How can I / How to
    m_how = re.match(r"(?i)^(?:how\s+(?:do\s+i|can\s+i|to))\s+(.+)", cleaned)
    if m_how:
        action = m_how.group(1).rstrip(".?! ")
        fallbacks.extend([
            f"What steps can I take to {action}?",
            f"Guide on how to {action}",
            f"Troubleshooting procedure to {action}",
            f"Can someone explain how to {action}?",
            f"Need instructions on how to {action}",
            f"Best way to {action}",
            f"What is the recommended method to {action}?",
            f"Steps required to {action}",
            f"How should I go about trying to {action}?",
            f"Troubleshooting steps to {action}",
            f"Where in settings can I {action}?",
            f"Assistance needed to {action}",
            f"How to {action}",
            f"Instructions to resolve: {action}",
        ])
        return fallbacks

    # Check for question starting with Why is / Why does / Why won't
    m_why = re.match(r"(?i)^(?:why\s+(?:is|does|won't|wont|cant|can't))\s+(.+)", cleaned)
    if m_why:
        issue = m_why.group(1).rstrip(".?! ")
        fallbacks.extend([
            f"What can I do if {issue}?",
            f"How do I fix when {issue}?",
            f"Troubleshooting steps for {issue}",
            f"What causes {issue} and how do I solve it?",
            f"How to resolve issue where {issue}",
            f"Assistance needed with {issue}",
            f"Steps to resolve when {issue}",
            f"Device troubleshooting help: {issue}",
            f"What should I do if {issue}?",
            f"Why is {issue} and how to fix it?",
            f"Looking for solutions when {issue}",
            f"Guide to fixing: {issue}",
            f"How to solve {issue}",
            f"Troubleshooting procedure for {issue}",
        ])
        return fallbacks

    # Standard symptom / problem statement
    if cleaned.lower().startswith("my "):
        symptom_clause = cleaned[0].lower() + cleaned[1:].rstrip(".?! ")
        core_phrase = cleaned[3:].strip().rstrip(".?! ")
    else:
        symptom_clause = cleaned[0].lower() + cleaned[1:].rstrip(".?! ")
        core_phrase = cleaned.rstrip(".?! ")

    fallbacks.extend([
        f"How can I fix: {core_phrase}?",
        f"What steps should I take if {symptom_clause}?",
        f"Troubleshooting steps when {symptom_clause}",
        f"How do I resolve this issue: {core_phrase}?",
        f"Device troubleshooting assistance needed: {core_phrase}",
        f"What can I do if {symptom_clause}?",
        f"How to troubleshoot: {core_phrase}",
        f"Steps to resolve: {core_phrase}",
        f"Guide to troubleshooting {symptom_clause}",
        f"What should I do when {symptom_clause}?",
        f"Can someone help fix this: {core_phrase}?",
        f"How do I fix when {symptom_clause}?",
        f"Looking for solutions for: {core_phrase}",
        f"Troubleshooting guide for when {symptom_clause}",
        f"Recommended fix if {symptom_clause}",
        f"How to solve problem where {symptom_clause}",
    ])

    return fallbacks


def _parse_gemini_variations(raw_text: str) -> List[str]:
    """Safely parse Gemini JSON text into a list of non-empty strings."""
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return [
                str(p).strip()
                for p in parsed
                if isinstance(p, (str, int, float)) and str(p).strip()
            ]
        elif isinstance(parsed, dict):
            for key in ("paraphrases", "variations", "queries"):
                if key in parsed and isinstance(parsed[key], list):
                    return [
                        str(p).strip()
                        for p in parsed[key]
                        if isinstance(p, (str, int, float)) and str(p).strip()
                    ]
        logger.warning(f"Unexpected JSON structure from Gemini: {parsed}")
        return []
    except Exception as e:
        logger.warning(f"Failed to parse JSON from Gemini response: {e}")
        return []


def _request_gemini(client: genai.Client, prompt: str) -> List[str]:
    """Execute Gemini request and return parsed variation strings."""
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return _parse_gemini_variations(response.text or "")
    except Exception as e:
        logger.warning(f"Gemini API request failed: {e}")
        return []


def generate_paraphrases(query: str, n: int = 9) -> List[str]:
    """Generate 8–10 unique, diverse linguistic paraphrases of a user query.

    Guarantees:
    - Deduplication: exact, case-insensitive, whitespace-collapsed, and against the original query.
    - Result count: strictly between 8 and 10 variations (capped at 10, minimum 8).
    - Robustness: if Gemini returns fewer than 8 usable variations, attempts one controlled
      retry. If still under target, supplements with deterministic, semantically faithful fallbacks.
    - Safety: preserves exact device model and symptoms without inventing technical facts or URLs.
    """
    clean_query = query.strip() if query else ""
    if not clean_query or len(clean_query) < 4 or len(clean_query.split()) < 2:
        raise ValueError(
            f"Cannot generate query variations: query '{query}' is empty or degenerate"
        )

    target_count = max(8, min(n, 10))

    # Maintain seen keys for case/whitespace-collapsed uniqueness
    seen_norm: Set[str] = set()
    # Seed original query so it can never be returned as a variation
    seen_norm.add(_normalize_for_comparison(clean_query))
    seen_norm.add(re.sub(r"\s+", " ", clean_query).lower())

    accepted: List[str] = []

    def _try_add(candidate: str) -> bool:
        c_clean = candidate.strip()
        if not c_clean:
            return False
        c_norm = _normalize_for_comparison(c_clean)
        c_lower = re.sub(r"\s+", " ", c_clean).lower()
        if c_norm in seen_norm or c_lower in seen_norm:
            return False
        seen_norm.add(c_norm)
        seen_norm.add(c_lower)
        accepted.append(c_clean)
        return True

    # 1. Attempt 1 via Gemini
    try:
        client = _get_genai_client()
        prompt_1 = (
            f"You are a semantic paraphrase generator for Samsung mobile device troubleshooting queries.\n"
            f"Generate exactly {target_count} natural linguistic variations of this user query.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. Keep the exact technical problem, device model, and key symptoms identical.\n"
            f"2. Rephrase sentence structure and tone naturally (formal, concise, conversational, casual).\n"
            f"3. Do NOT provide solutions, advice, or troubleshooting steps.\n"
            f"4. Do NOT repeat or return the original query.\n"
            f"5. Output ONLY a valid JSON list of strings containing exactly {target_count} paraphrases:\n"
            f'["paraphrase 1", "paraphrase 2", ...]\n\n'
            f"Original Query:\n"
            f'"{clean_query}"\n'
        )
        for item in _request_gemini(client, prompt_1):
            if len(accepted) >= target_count:
                break
            _try_add(item)
    except Exception as e:
        logger.warning(f"Initial Gemini generation failed: {e}")

    # 2. Controlled Retry if fewer than 8 variations obtained
    if len(accepted) < 8:
        try:
            client = _get_genai_client()
            needed = target_count - len(accepted)
            prompt_2 = (
                f"You are a semantic paraphrase generator for Samsung mobile device troubleshooting queries.\n"
                f"Generate exactly {needed + 4} ADDITIONAL distinct linguistic variations of this user query.\n\n"
                f"CRITICAL REQUIREMENTS:\n"
                f"1. Keep the exact technical problem, device model, and symptoms identical.\n"
                f"2. Substantially vary sentence openings and grammar structures.\n"
                f"3. Do NOT provide solutions or advice.\n"
                f"4. Do NOT repeat the original query.\n"
                f"5. Output ONLY a valid JSON list of strings:\n"
                f'["variation 1", "variation 2", ...]\n\n'
                f"Original Query:\n"
                f'"{clean_query}"\n'
            )
            for item in _request_gemini(client, prompt_2):
                if len(accepted) >= target_count:
                    break
                _try_add(item)
        except Exception as e:
            logger.warning(f"Controlled retry failed: {e}")

    # 3. Deterministic rule-based fallback if still under target_count
    if len(accepted) < target_count:
        for fallback in _generate_deterministic_fallbacks(clean_query):
            if len(accepted) >= target_count:
                break
            _try_add(fallback)

    return accepted[:target_count]
