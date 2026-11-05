import logging
import os
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.cache import compute_siis_hash, lookup, store
from app.extraction import extract_goal
from app.schema import ContextDeeplinkResponse, ResponseMeta
from app.validators import validate_full_response

logger = logging.getLogger(__name__)

app = FastAPI(title="TapFix", version="0.1.0")


class SIISContent(BaseModel):
    title: str
    content: str


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: SIISContent


_CACHE_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.82"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/troubleshoot", response_model=ContextDeeplinkResponse)
async def troubleshoot(req: TroubleshootRequest):
    """POST /v1/troubleshoot with Semantic Cache write-through architecture.

    Request Flow:
    1. Start latency timer.
    2. Compute deterministic SIIS hash from req.siis_response.
    3. Lookup query in semantic cache scoped by SIIS hash (threshold=0.82).
    4. On CACHE HIT:
       - return cached validated response immediately (without calling Gemini)
       - set meta.cache_hit = True
       - record latency_ms
    5. On CACHE MISS:
       - execute existing Phase 3 extraction pipeline unchanged
       - validate full response
       - if valid and completed without errors, store validated response in cache with SIIS hash
       - set meta.cache_hit = False
       - record latency_ms
       - return response
    """
    t0 = time.perf_counter()

    # Deterministic SIIS identity hash
    siis_hash = compute_siis_hash(req.siis_response.title, req.siis_response.content)

    # 1. Semantic Cache Lookup (scoped to both query and SIIS context)
    cached_data: Optional[dict] = None
    try:
        cached_data = lookup(req.query, threshold=_CACHE_THRESHOLD, siis_hash=siis_hash)
    except Exception as e:
        logger.error(f"Semantic cache lookup error (continuing with cold path): {e}")
        cached_data = None

    # CACHE HIT
    if cached_data is not None:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        resp = ContextDeeplinkResponse(**cached_data)
        resp.meta = ResponseMeta(cache_hit=True, latency_ms=latency_ms)
        return resp

    # CACHE MISS: Run existing Phase 3 pipeline unchanged
    result = extract_goal(req.query, req.siis_response.title, req.siis_response.content)

    resp_tree = ContextDeeplinkResponse(**result)
    is_valid, failures = validate_full_response(resp_tree)

    # Cache write-through: store only successfully validated responses
    meta_info = result.get("_meta", {})
    final_valid = meta_info.get("final_valid", True)
    has_error = "error" in meta_info or "fallback" in result

    if is_valid and final_valid and not has_error:
        try:
            clean_to_cache = resp_tree.model_dump(exclude={"meta"})
            store(req.query, None, clean_to_cache, siis_hash=siis_hash)
        except Exception as e:
            logger.error(f"Semantic cache store error: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    resp_tree.meta = ResponseMeta(cache_hit=False, latency_ms=latency_ms)
    return resp_tree

