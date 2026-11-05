import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.extraction import extract_goal
from app.schema import ContextDeeplinkResponse, actionCategory
from app.validators import scrub_urls, validate_full_response

_ROOT = Path(__file__).resolve().parent.parent
_SIIS_PATH = _ROOT / "data" / "siis_responses.json"
_RESULTS_PATH = _ROOT / "results.jsonl"


def run_benchmark():
    with open(_SIIS_PATH, "r", encoding="utf-8") as f:
        siis_data = json.load(f)

    responses = siis_data.get("responses", [])
    total_queries = len(responses)
    print(f"Starting Phase 3 Benchmark on {total_queries} queries...\n")

    results_file = open(_RESULTS_PATH, "w", encoding="utf-8")

    first_attempt_pass = 0
    retry_pass = 0
    failed_count = 0
    url_leak_count = 0
    dummy_positive_count = 0
    auto_actions_total = 0
    auto_actions_with_deeplink = 0
    total_steps = 0

    start_time = time.time()

    for idx, item in enumerate(responses, 1):
        query_id = item.get("id", f"row_{idx}")
        query = item.get("original_query", "")
        siis = item.get("siis_response", {})
        title = siis.get("title", "")
        content = siis.get("content", "")

        print(f"[{idx}/{total_queries}] Processing {query_id}...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = extract_goal(query, title, content)
        except Exception as e:
            duration = time.time() - t0
            print(f"FAIL: exception ({duration:.1f}s)")
            print(f"  Error: {e}", file=sys.stderr)
            failed_count += 1
            record = {
                "id": query_id,
                "query": query,
                "duration_seconds": round(duration, 2),
                "is_valid": False,
                "first_attempt_valid": False,
                "retry_performed": False,
                "validation_failures": [f"Unhandled exception: {e}"],
                "response": {"contexts": []},
            }
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            continue

        duration = time.time() - t0

        meta = result.get("_meta", {})
        first_valid = meta.get("first_attempt_valid", False)
        retry_done = meta.get("retry_performed", False)
        final_valid = meta.get("final_valid", False)
        had_error = "error" in meta

        resp_tree = ContextDeeplinkResponse(**result)
        is_schema_valid, failures = validate_full_response(resp_tree)

        # Count URL leaks & auto deeplinks & dummy positive occurrences
        query_url_leaks = 0
        query_dummy_positives = 0
        query_auto_actions = 0
        query_auto_with_dl = 0

        for goal in resp_tree.contexts:
            for action in goal.actions:
                is_auto_or_crit = action.category in [actionCategory.auto, actionCategory.critical]
                for sg in action.stepGroups:
                    for s in sg.steps:
                        total_steps += 1
                        if scrub_urls(s) != s:
                            query_url_leaks += 1
                    if sg.actionableDeeplink is not None and "dummy_positive" in str(sg.actionableDeeplink.deeplink):
                        query_dummy_positives += 1
                    if sg.validationDeeplink is not None and "dummy_positive" in str(sg.validationDeeplink.deeplink):
                        query_dummy_positives += 1
                    if is_auto_or_crit:
                        query_auto_actions += 1
                        if sg.actionableDeeplink is not None and sg.actionableDeeplink.deeplink:
                            query_auto_with_dl += 1

        url_leak_count += query_url_leaks
        dummy_positive_count += query_dummy_positives
        auto_actions_total += query_auto_actions
        auto_actions_with_deeplink += query_auto_with_dl

        # Determine actual validity: schema must pass AND pipeline must not have errored
        actually_valid = is_schema_valid and final_valid and not had_error

        if actually_valid:
            if first_valid:
                first_attempt_pass += 1
                status = "PASS (1st try)"
            elif retry_done:
                retry_pass += 1
                status = "PASS (retry)"
            else:
                first_attempt_pass += 1
                status = "PASS"
        else:
            failed_count += 1
            if had_error:
                status = f"FAIL: {meta.get('error', 'unknown error')}"
            elif not is_schema_valid:
                status = f"FAIL: {failures}"
            else:
                status = f"FAIL: final_valid=False, failures={meta.get('failures', [])}"

        print(f"{status} ({duration:.1f}s)")

        # Write to results.jsonl
        record = {
            "id": query_id,
            "query": query,
            "duration_seconds": round(duration, 2),
            "is_valid": actually_valid,
            "first_attempt_valid": first_valid and is_schema_valid and not had_error,
            "retry_performed": retry_done,
            "validation_failures": failures if not is_schema_valid else meta.get("failures", []),
            "response": {
                "contexts": result.get("contexts", [])
            },
        }
        results_file.write(json.dumps(record) + "\n")
        results_file.flush()

    results_file.close()
    elapsed = time.time() - start_time

    total_valid = first_attempt_pass + retry_pass

    print("\n" + "=" * 60)
    print("PHASE 3 BENCHMARK REPORT")
    print("=" * 60)
    print(f"Total queries processed:       {total_queries}")
    print(f"Valid on first attempt:        {first_attempt_pass}/{total_queries}")
    print(f"Valid after single retry:      {retry_pass}/{total_queries}")
    print(f"Total valid outputs:           {total_valid}/{total_queries} (Target: >= 18/20)")
    print(f"Failed both attempts:          {failed_count}/{total_queries}")
    print(f"Total steps evaluated:         {total_steps}")
    print(f"URL leaks detected:            {url_leak_count} (Target: 0)")
    print(f"Dummy positive occurrences:    {dummy_positive_count} (Target: 0)")
    auto_coverage = (
        (auto_actions_with_deeplink / auto_actions_total * 100)
        if auto_actions_total > 0
        else 100.0
    )
    print(f"Auto/critical deeplink coverage: {auto_actions_with_deeplink}/{auto_actions_total} ({auto_coverage:.1f}%) (Target: 100%)")
    print(f"Total elapsed time:            {elapsed:.1f}s")
    print(f"Results written to:            {_RESULTS_PATH}")
    print("=" * 60)

    if total_valid < 18:
        print("\nWARNING: Benchmark target not met (<18 valid outputs).")
        return 1
    if url_leak_count > 0:
        print("\nWARNING: Benchmark target not met (URL leaks > 0).")
        return 1
    if dummy_positive_count > 0:
        print("\nWARNING: Benchmark target not met (dummy_positive occurrences > 0).")
        return 1
    if auto_actions_with_deeplink < auto_actions_total:
        print("\nWARNING: Benchmark target not met (auto actions missing actionableDeeplink).")
        return 1

    print("\nALL PHASE 3 CHECKPOINT CRITERIA MET SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
