import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.cache import clear_cache, lookup_with_score
from app.deeplink_match import _model
from app.enrichment import generate_paraphrases
from app.main import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_BENCHMARK_ROW_IDS = ["row_2", "row_4", "row_13", "row_16", "row_21"]


def run_cache_benchmark() -> Dict[str, Any]:
    siis_path = _ROOT / "data" / "siis_responses.json"
    with open(siis_path, "r", encoding="utf-8") as f:
        siis_data = json.load(f)

    all_responses = {r["id"]: r for r in siis_data.get("responses", [])}

    client = TestClient(app)

    print("=" * 70)
    print("PHASE 4 — SEMANTIC CACHE BENCHMARK (5 QUERIES × 5 PARAPHRASES)")
    print("=" * 70)

    per_query_results = []
    all_cold_latencies: List[float] = []
    all_cache_hit_latencies: List[float] = []
    total_paraphrases = 0
    total_hits = 0

    for idx, row_id in enumerate(_BENCHMARK_ROW_IDS, 1):
        item = all_responses.get(row_id)
        if not item:
            print(f"Error: {row_id} not found in siis_responses.json")
            continue

        original_query = item["original_query"]
        siis_title = item["siis_response"]["title"]
        siis_content = item["siis_response"]["content"]

        print(f"\n[{idx}/5] Evaluating {row_id}:")
        print(f"  Original Query: \"{original_query[:80]}...\"")

        # Step 1: Clear the cache for clean isolation
        clear_cache()

        # Step 2: Run original query as a cold request
        payload = {
            "query": original_query,
            "siis_response": {
                "title": siis_title,
                "content": siis_content,
            },
        }

        resp_cold = client.post("/v1/troubleshoot", json=payload)
        if resp_cold.status_code != 200:
            print(f"  Cold request failed with status {resp_cold.status_code}")
            continue

        cold_data = resp_cold.json()
        cold_meta = cold_data.get("meta", {})
        cold_latency_ms = cold_meta.get("latency_ms", 0.0)
        is_cold = not cold_meta.get("cache_hit", False)

        all_cold_latencies.append(cold_latency_ms)
        print(f"  Cold Request: {cold_latency_ms:.2f} ms (cache_hit={not is_cold})")

        # Step 3: Generate 5 paraphrases using Gemini
        print("  Generating 5 diverse paraphrases via Gemini...")
        paraphrases = generate_paraphrases(original_query, n=5)
        if len(paraphrases) < 5:
            # Fallback retry to ensure 5 paraphrases
            extra = generate_paraphrases(original_query, n=5 - len(paraphrases))
            paraphrases.extend(extra)
        paraphrases = paraphrases[:5]

        # Step 4: Run all 5 paraphrases
        query_emb = _model.encode(original_query, normalize_embeddings=True)
        paraphrase_records = []
        query_hits = 0

        for p_idx, para in enumerate(paraphrases, 1):
            para_emb = _model.encode(para, normalize_embeddings=True)
            sim = float(np.dot(query_emb, para_emb))

            para_payload = {
                "query": para,
                "siis_response": {
                    "title": siis_title,
                    "content": siis_content,
                },
            }

            resp_p = client.post("/v1/troubleshoot", json=para_payload)
            p_data = resp_p.json()
            p_meta = p_data.get("meta", {})
            hit = p_meta.get("cache_hit", False)
            lat_ms = p_meta.get("latency_ms", 0.0)

            if hit:
                query_hits += 1
                total_hits += 1
                all_cache_hit_latencies.append(lat_ms)

            total_paraphrases += 1
            status_str = "HIT " if hit else "MISS"
            print(f"    [{p_idx}/5] {status_str} (sim={sim:.4f}, {lat_ms:.2f}ms): \"{para[:65]}...\"")

            paraphrase_records.append({
                "paraphrase": para,
                "similarity": round(sim, 4),
                "cache_hit": hit,
                "latency_ms": lat_ms,
            })

        hit_rate = (query_hits / len(paraphrases) * 100) if paraphrases else 0.0
        per_query_results.append({
            "row_id": row_id,
            "cold_latency_ms": cold_latency_ms,
            "hits": query_hits,
            "total_paraphrases": len(paraphrases),
            "hit_rate_pct": round(hit_rate, 1),
            "paraphrases": paraphrase_records,
        })

    # Step 5: Report Aggregate Metrics
    agg_hit_rate = (total_hits / total_paraphrases * 100) if total_paraphrases > 0 else 0.0
    cold_p50 = float(np.percentile(all_cold_latencies, 50)) if all_cold_latencies else 0.0
    cold_p95 = float(np.percentile(all_cold_latencies, 95)) if all_cold_latencies else 0.0
    hit_p50 = float(np.percentile(all_cache_hit_latencies, 50)) if all_cache_hit_latencies else 0.0
    hit_p95 = float(np.percentile(all_cache_hit_latencies, 95)) if all_cache_hit_latencies else 0.0

    print("\n" + "=" * 70)
    print("PHASE 4 BENCHMARK REPORT — SUMMARY")
    print("=" * 70)
    print(f"{'Query ID':<10} | {'Cold Latency':<14} | {'Hits / Total':<14} | {'Hit Rate':<10}")
    print("-" * 56)
    for q in per_query_results:
        print(f"{q['row_id']:<10} | {q['cold_latency_ms']:>9.2f} ms   | {q['hits']}/{q['total_paraphrases']:<12} | {q['hit_rate_pct']:>6.1f} %")
    print("-" * 56)
    print(f"Total Paraphrases Evaluated: {total_paraphrases}")
    print(f"Total Cache Hits:            {total_hits}")
    print(f"Paraphrase Cache-Hit Rate:   {agg_hit_rate:.1f}% (Target: >= 80.0%)")
    print(f"Cold Latency P50:            {cold_p50:.2f} ms")
    print(f"Cold Latency P95:            {cold_p95:.2f} ms (Target: <= 8000.0 ms)")
    print(f"Cache-Hit Latency P50:       {hit_p50:.2f} ms")
    print(f"Cache-Hit Latency P95:       {hit_p95:.2f} ms (Target: <= 300.0 ms)")
    print("=" * 70)

    # Checkpoint Validation
    pass_hit_rate = agg_hit_rate >= 80.0
    pass_cold_p95 = cold_p95 <= 8000.0
    pass_hit_p95 = hit_p95 <= 300.0

    if pass_hit_rate and pass_cold_p95 and pass_hit_p95:
        print("\nALL PHASE 4 CHECKPOINT CRITERIA MET SUCCESSFULLY!\n")
    else:
        print("\nWARNING: Some Phase 4 Checkpoint criteria were not met:")
        if not pass_hit_rate:
            print(f"  - Hit rate below target: {agg_hit_rate:.1f}% < 80.0%")
        if not pass_cold_p95:
            print(f"  - Cold P95 exceeded target: {cold_p95:.2f} ms > 8000.0 ms")
        if not pass_hit_p95:
            print(f"  - Cache-Hit P95 exceeded target: {hit_p95:.2f} ms > 300.0 ms")

    return {
        "per_query": per_query_results,
        "total_paraphrases": total_paraphrases,
        "total_hits": total_hits,
        "hit_rate_pct": round(agg_hit_rate, 2),
        "cold_p50_ms": round(cold_p50, 2),
        "cold_p95_ms": round(cold_p95, 2),
        "cache_hit_p50_ms": round(hit_p50, 2),
        "cache_hit_p95_ms": round(hit_p95, 2),
        "checkpoint_passed": pass_hit_rate and pass_cold_p95 and pass_hit_p95,
    }


if __name__ == "__main__":
    run_cache_benchmark()
