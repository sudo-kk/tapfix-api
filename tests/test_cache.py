import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache import (
    clear_cache,
    compute_siis_hash,
    init_db,
    lookup,
    lookup_with_score,
    store,
)
from app.enrichment import generate_paraphrases
from app.main import app
from app.schema import (
    Action,
    ContextDeeplinkResponse,
    Goal,
    StepGroup,
    actionCategory,
)

_VALID_TEST_RESPONSE = {
    "contexts": [
        {
            "goal": "Follow these steps to perform this Blank Display Troubleshooting",
            "title": "Blank display issue",
            "score": 0.95,
            "actions": [
                {
                    "actionName": "Force Restart Device",
                    "description": "It will force restart your mobile device",
                    "category": "manual",
                    "stepGroups": [
                        {
                            "steps": [
                                "Press and hold the Power and Volume down buttons for 20 seconds."
                            ],
                            "actionableDeeplink": None,
                            "validationDeeplink": None,
                        }
                    ],
                }
            ],
        }
    ]
}


def test_empty_cache_lookup(tmp_path):
    """Test that looking up any query in an empty cache safely returns None."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    result = lookup("My phone display is completely black", threshold=0.85, db_path=db_file)
    assert result is None


def test_store_and_lookup_exact(tmp_path):
    """Test that storing a query and response allows exact-query lookup."""
    db_file = tmp_path / "test_cache.db"
    query = "My Galaxy S22 screen turns completely blank or white and no text appears"
    store(query, None, _VALID_TEST_RESPONSE, db_path=db_file)

    cached = lookup(query, threshold=0.85, db_path=db_file)
    assert cached is not None
    assert cached["contexts"][0]["title"] == "Blank display issue"


def test_semantic_paraphrase_lookup(tmp_path):
    """Test that semantically similar paraphrases successfully hit the cache."""
    db_file = tmp_path / "test_cache.db"
    canonical = "My Galaxy S22 screen turns completely blank or white and no text appears"
    store(canonical, None, _VALID_TEST_RESPONSE, db_path=db_file)

    paraphrase = "My Galaxy S22 display goes fully white or blank without showing text"
    cached, score = lookup_with_score(paraphrase, threshold=0.85, db_path=db_file)
    assert cached is not None
    assert score >= 0.85
    assert cached["contexts"][0]["title"] == "Blank display issue"


def test_below_threshold_miss(tmp_path):
    """Test that semantically unrelated queries do not trigger false positive cache hits."""
    db_file = tmp_path / "test_cache.db"
    canonical = "My Galaxy S22 screen turns completely blank or white and no text appears"
    store(canonical, None, _VALID_TEST_RESPONSE, db_path=db_file)

    unrelated = "How do I configure my exchange email server and sync calendar events?"
    cached, score = lookup_with_score(unrelated, threshold=0.85, db_path=db_file)
    assert cached is None
    assert score < 0.85


def test_corrupted_embedding_handling(tmp_path):
    """Test that corrupted/invalid embeddings in SQLite do not crash lookup and are ignored safely."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)

    # 1. Insert a row with corrupted binary data (e.g. truncated / wrong length)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            "INSERT INTO cache (canonical_query, embedding, response_json, created_at) VALUES (?, ?, ?, ?)",
            ("Corrupt row 1", b"bad_truncated_data", json.dumps(_VALID_TEST_RESPONSE), "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO cache (canonical_query, embedding, response_json, created_at) VALUES (?, ?, ?, ?)",
            ("Corrupt row 2 (wrong dim)", np.zeros(100, dtype=np.float32).tobytes(), json.dumps(_VALID_TEST_RESPONSE), "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    # 2. Lookup on corrupted db must not raise exception
    res = lookup("Any random query", threshold=0.85, db_path=db_file)
    assert res is None

    # 3. Store a valid row alongside corrupt ones
    valid_query = "Galaxy S22 blank white screen"
    store(valid_query, None, _VALID_TEST_RESPONSE, db_path=db_file)

    # Valid query must still be retrievable without being blocked by corrupt rows
    hit = lookup(valid_query, threshold=0.85, db_path=db_file)
    assert hit is not None
    assert hit["contexts"][0]["title"] == "Blank display issue"


def test_cache_hit_avoids_cold_path_execution(tmp_path, monkeypatch):
    """Verify that on CACHE HIT, the response returns immediately without calling extract_goal."""
    db_file = tmp_path / "test_cache.db"
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    query = "My Galaxy S22 screen is black"
    req_payload = {
        "query": query,
        "siis_response": {
            "title": "Blank or black display on a Samsung phone or tablet",
            "content": "Troubleshooting steps...",
        },
    }
    siis_h = compute_siis_hash(req_payload["siis_response"]["title"], req_payload["siis_response"]["content"])
    store(query, None, _VALID_TEST_RESPONSE, db_path=db_file, siis_hash=siis_h)

    mock_extract = MagicMock()
    monkeypatch.setattr("app.main.extract_goal", mock_extract)

    client = TestClient(app)
    response = client.post("/v1/troubleshoot", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["cache_hit"] is True
    assert data["meta"]["latency_ms"] >= 0.0
    mock_extract.assert_not_called()


def test_cache_miss_stores_validated_response(tmp_path, monkeypatch):
    """Verify that on CACHE MISS, a valid response from the pipeline is saved to semantic cache."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    query = "New query that is not yet in the cache"
    mock_result = dict(_VALID_TEST_RESPONSE)
    mock_result["_meta"] = {"first_attempt_valid": True, "final_valid": True, "failures": []}

    monkeypatch.setattr("app.main.extract_goal", lambda q, t, c: mock_result)

    client = TestClient(app)
    req_payload = {
        "query": query,
        "siis_response": {
            "title": "Blank display issue",
            "content": "Steps...",
        },
    }

    response = client.post("/v1/troubleshoot", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["cache_hit"] is False

    # Now verify the response was cached
    siis_h = compute_siis_hash(req_payload["siis_response"]["title"], req_payload["siis_response"]["content"])
    cached = lookup(query, threshold=0.85, db_path=db_file, siis_hash=siis_h)
    assert cached is not None
    assert cached["contexts"][0]["title"] == "Blank display issue"


def test_failed_or_invalid_response_is_not_cached(tmp_path, monkeypatch):
    """Verify that validation failures or pipeline error objects are NEVER stored into cache."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    query = "Query with validation failure"
    # Result with invalid description format (not starting with 'It will') and fallback
    invalid_result = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Bad Format",
                "title": "Invalid title words count is four",
                "score": 0.5,
                "actions": [],
            }
        ],
        "fallback": "validation_failed",
        "_meta": {"final_valid": False, "error": "Validation error"},
    }

    monkeypatch.setattr("app.main.extract_goal", lambda q, t, c: invalid_result)

    client = TestClient(app)
    req_payload = {
        "query": query,
        "siis_response": {
            "title": "Test Title",
            "content": "Test Content",
        },
    }

    response = client.post("/v1/troubleshoot", json=req_payload)
    assert response.status_code == 200

    # Ensure nothing was stored in cache
    siis_h = compute_siis_hash(req_payload["siis_response"]["title"], req_payload["siis_response"]["content"])
    cached = lookup(query, threshold=0.85, db_path=db_file, siis_hash=siis_h)
    assert cached is None


def test_cache_latency_metadata(tmp_path, monkeypatch):
    """Verify that both cache hits and misses include proper latency_ms float and cache_hit boolean."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    mock_result = dict(_VALID_TEST_RESPONSE)
    mock_result["_meta"] = {"first_attempt_valid": True, "final_valid": True, "failures": []}
    monkeypatch.setattr("app.main.extract_goal", lambda q, t, c: mock_result)

    client = TestClient(app)
    req = {
        "query": "Sample latency query",
        "siis_response": {"title": "Title", "content": "Content"},
    }

    # Miss
    resp1 = client.post("/v1/troubleshoot", json=req)
    assert resp1.status_code == 200
    meta1 = resp1.json()["meta"]
    assert meta1["cache_hit"] is False
    assert isinstance(meta1["latency_ms"], (int, float))
    assert meta1["latency_ms"] >= 0.0

    # Hit
    resp2 = client.post("/v1/troubleshoot", json=req)
    assert resp2.status_code == 200
    meta2 = resp2.json()["meta"]
    assert meta2["cache_hit"] is True
    assert isinstance(meta2["latency_ms"], (int, float))
    assert meta2["latency_ms"] >= 0.0


def test_paraphrase_generation_output_shape(monkeypatch):
    """Verify generate_paraphrases returns expected list of strings with correct shape."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps([
        "Galaxy screen is flashing when charger is plugged in",
        "Screen blinks rapidly whenever connected to charging cable",
        "Display flickers violently on my Samsung during charging",
        "My phone screen flashes real fast when on charger plz fix",
        "Plugging in charger makes the screen rapidly flash on my phone"
    ])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp
    monkeypatch.setattr("app.enrichment._get_genai_client", lambda: mock_client)

    paraphrases = generate_paraphrases("My phone screen flashes when plugged in", n=5)
    assert isinstance(paraphrases, list)
    assert 8 <= len(paraphrases) <= 10
    for p in paraphrases:
        assert isinstance(p, str)
        assert len(p) > 5


# ==============================================================================
# Phase 6 Regression Tests: SIIS Scoped Semantic Cache
# ==============================================================================

def test_cache_scope_case_a_same_query_same_siis(tmp_path, monkeypatch):
    """Case A: same query + same SIIS -> cache hit on second request with identical response."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    mock_extract = MagicMock()
    mock_result = dict(_VALID_TEST_RESPONSE)
    mock_result["_meta"] = {"first_attempt_valid": True, "final_valid": True, "failures": []}
    mock_extract.return_value = mock_result
    monkeypatch.setattr("app.main.extract_goal", mock_extract)

    client = TestClient(app)
    payload = {
        "query": "My phone screen is having issues and I cannot see anything clearly.",
        "siis_response": {
            "title": "Screen and Display Settings Troubleshooting",
            "content": "1. Enable Touch Sensitivity: Open Settings.\n2. Clean the screen.",
        },
    }

    # Request 1: Cold path -> cache_hit=False
    resp1 = client.post("/v1/troubleshoot", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["meta"]["cache_hit"] is False
    assert mock_extract.call_count == 1

    # Request 2: Warm path -> cache_hit=True
    resp2 = client.post("/v1/troubleshoot", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["meta"]["cache_hit"] is True
    # extract_goal was not called again
    assert mock_extract.call_count == 1
    # Responses match
    assert data1["contexts"] == data2["contexts"]


def test_cache_scope_case_b_same_query_different_siis(tmp_path, monkeypatch):
    """Case B: same query + materially DIFFERENT SIIS -> MUST NOT return cached response from first SIIS."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.85, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    siis_1_resp = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "Display issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Enable Touch Sensitivity",
                        "description": "It will enable touchscreen sensitivity settings",
                        "category": "auto",
                        "stepGroups": [{"steps": ["Open Settings", "Tap Display"]}],
                    }
                ],
            }
        ],
        "_meta": {"first_attempt_valid": True, "final_valid": True, "failures": []},
    }

    siis_2_resp = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Hardware Troubleshooting",
                "title": "Hardware damage issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Force Restart Device",
                        "description": "It will force restart your mobile device",
                        "category": "manual",
                        "stepGroups": [{"steps": ["Hold Power and Volume down"]}],
                    }
                ],
            }
        ],
        "_meta": {"first_attempt_valid": True, "final_valid": True, "failures": []},
    }

    def mock_extract(q, t, c):
        if "Hardware" in t or "Blank Screen" in t:
            return siis_2_resp
        return siis_1_resp

    monkeypatch.setattr("app.main.extract_goal", mock_extract)

    client = TestClient(app)
    common_query = "My phone screen is having issues and I cannot see anything clearly."

    payload_1 = {
        "query": common_query,
        "siis_response": {
            "title": "Screen and Display Settings Troubleshooting",
            "content": "1. Enable Touch Sensitivity: Open Settings.\n2. Clean the screen.",
        },
    }

    payload_2 = {
        "query": common_query,
        "siis_response": {
            "title": "Blank Screen Hardware Troubleshooting",
            "content": "1. Inspect Hardware: check cables.\n2. Force Restart Phone: press buttons.",
        },
    }

    # Request 1: SIIS 1 -> Populates cache with SIIS 1 response
    resp1 = client.post("/v1/troubleshoot", json=payload_1)
    assert resp1.status_code == 200
    assert resp1.json()["meta"]["cache_hit"] is False
    actions1 = [a["actionName"] for a in resp1.json()["contexts"][0]["actions"]]
    assert actions1 == ["Enable Touch Sensitivity"]

    # Request 2: SIIS 2 with SAME query -> MUST be cache miss and return SIIS 2 response!
    resp2 = client.post("/v1/troubleshoot", json=payload_2)
    assert resp2.status_code == 200
    assert resp2.json()["meta"]["cache_hit"] is False  # Must NOT hit SIIS 1 cache!
    actions2 = [a["actionName"] for a in resp2.json()["contexts"][0]["actions"]]
    assert actions2 == ["Force Restart Device"]
    assert "Enable Touch Sensitivity" not in actions2


def test_cache_scope_case_c_paraphrase_same_siis(tmp_path, monkeypatch):
    """Case C: Semantically similar / paraphrased query + same SIIS -> cache hit."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.82, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    mock_extract = MagicMock()
    mock_result = dict(_VALID_TEST_RESPONSE)
    mock_result["_meta"] = {"first_attempt_valid": True, "final_valid": True, "failures": []}
    mock_extract.return_value = mock_result
    monkeypatch.setattr("app.main.extract_goal", mock_extract)

    client = TestClient(app)
    siis_doc = {
        "title": "Blank or black display on a Samsung phone or tablet",
        "content": "Troubleshooting steps for blank or black screen issues...",
    }

    # Request 1 with base query
    resp1 = client.post(
        "/v1/troubleshoot",
        json={
            "query": "My Galaxy S22 screen turns completely blank or white and no text appears",
            "siis_response": siis_doc,
        },
    )
    assert resp1.status_code == 200
    assert resp1.json()["meta"]["cache_hit"] is False
    assert mock_extract.call_count == 1

    # Request 2 with paraphrase + same SIIS -> should hit cache!
    resp2 = client.post(
        "/v1/troubleshoot",
        json={
            "query": "My Galaxy S22 display goes fully white or blank without showing text",
            "siis_response": siis_doc,
        },
    )
    assert resp2.status_code == 200
    assert resp2.json()["meta"]["cache_hit"] is True
    assert mock_extract.call_count == 1  # extract_goal was bypassed


def test_cache_scope_case_d_paraphrase_different_siis(tmp_path, monkeypatch):
    """Case D: Semantically similar / paraphrased query + different SIIS -> MUST NOT hit cache."""
    db_file = tmp_path / "test_cache.db"
    init_db(db_file)
    monkeypatch.setattr("app.main.lookup", lambda q, threshold=0.82, **kwargs: lookup(q, threshold, db_path=db_file, **kwargs))
    monkeypatch.setattr("app.main.store", lambda q, emb, resp, **kwargs: store(q, emb, resp, db_path=db_file, **kwargs))

    resp_display = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "Display issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Inspect Physical Screen",
                        "description": "It will inspect physical device display screen",
                        "category": "manual",
                        "stepGroups": [{"steps": ["Inspect screen"]}],
                    }
                ],
            }
        ],
        "_meta": {"first_attempt_valid": True, "final_valid": True, "failures": []},
    }

    resp_audio = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Audio Troubleshooting",
                "title": "Audio sound issue",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Inspect Audio Hardware",
                        "description": "It will inspect physical device audio hardware",
                        "category": "manual",
                        "stepGroups": [{"steps": ["Inspect speakers"]}],
                    }
                ],
            }
        ],
        "_meta": {"first_attempt_valid": True, "final_valid": True, "failures": []},
    }

    def mock_extract(q, t, c):
        if "Audio" in t:
            return resp_audio
        return resp_display

    monkeypatch.setattr("app.main.extract_goal", mock_extract)

    client = TestClient(app)

    # Request 1 with base query and Display SIIS
    resp1 = client.post(
        "/v1/troubleshoot",
        json={
            "query": "My Galaxy S22 screen turns completely blank or white and no text appears",
            "siis_response": {
                "title": "Display Troubleshooting Guide",
                "content": "Display content...",
            },
        },
    )
    assert resp1.status_code == 200
    assert resp1.json()["meta"]["cache_hit"] is False

    # Request 2 with paraphrase query and DIFFERENT (Audio) SIIS
    resp2 = client.post(
        "/v1/troubleshoot",
        json={
            "query": "My Galaxy S22 display goes fully white or blank without showing text",
            "siis_response": {
                "title": "Audio Troubleshooting Guide",
                "content": "Audio content...",
            },
        },
    )
    assert resp2.status_code == 200
    # Must be cache miss!
    assert resp2.json()["meta"]["cache_hit"] is False
    assert resp2.json()["contexts"][0]["title"] == "Audio sound issue"


def test_siis_hash_stability_across_process_and_invocations():
    """Verify compute_siis_hash is deterministic, stable, and handles whitespace/line-endings."""
    title = "Blank or black display on a Samsung phone or tablet"
    content = "Inspect phone, charger, and cable for damage.\r\nTurn off your phone.\r\n"

    h1 = compute_siis_hash(title, content)
    h2 = compute_siis_hash(title, content)
    # Determinism
    assert h1 == h2
    assert len(h1) == 64
    assert h1 == h1.lower()

    # Normalization: CRLF vs LF
    content_lf = "Inspect phone, charger, and cable for damage.\nTurn off your phone."
    h3 = compute_siis_hash(title, content_lf)
    assert h1 == h3

    # Leading/trailing whitespace normalization
    h4 = compute_siis_hash(f"  {title}  ", f"\n\n{content}\n  ")
    assert h1 == h4

    # Material difference must change hash
    h_diff = compute_siis_hash(title, content + " Additional step.")
    assert h1 != h_diff

    # Empty inputs
    assert compute_siis_hash("", "") == ""
    assert compute_siis_hash(None, None) == ""


def test_legacy_cache_rows_cannot_cause_cross_siis_contamination(tmp_path):
    """Verify that legacy SQLite records without siis_hash cannot match requests with a siis_hash."""
    db_file = tmp_path / "legacy_cache.db"

    # 1. Create a legacy table without the siis_hash column
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            """
            CREATE TABLE cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_query TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Insert a legacy row with dummy embedding
        dummy_emb = np.zeros(384, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO cache (canonical_query, embedding, response_json, created_at) VALUES (?, ?, ?, ?)",
            ("My screen is completely black", dummy_emb, json.dumps(_VALID_TEST_RESPONSE), "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    # 2. Running init_db on the legacy DB migrates the schema safely
    init_db(db_file)

    # Verify column was added
    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(cache)")
        cols = [r[1] for r in cursor.fetchall()]
        assert "siis_hash" in cols

    # 3. Looking up with a modern request (providing siis_hash) MUST NOT match the legacy row!
    modern_siis_hash = compute_siis_hash("Display Issue", "Steps...")
    hit, score = lookup_with_score(
        "My screen is completely black", threshold=0.5, db_path=db_file, siis_hash=modern_siis_hash
    )
    assert hit is None
    assert score == 0.0

