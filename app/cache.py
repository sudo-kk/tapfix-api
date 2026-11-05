import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from app.deeplink_match import _model

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "cache.db"
_EMBEDDING_DIM = 384


def compute_siis_hash(title: Optional[str] = None, content: Optional[str] = None) -> str:
    """Compute a deterministic, stable SHA-256 hash for SIIS title and content.

    Canonical representation:
    - title stripped of leading/trailing whitespace
    - content stripped of leading/trailing whitespace with CRLF normalized to LF
    - joined by newline: f"{norm_title}\\n{norm_content}"

    Returns a 64-character lowercase hexadecimal SHA-256 digest, or "" if both are empty.
    """
    norm_title = (title or "").strip()
    norm_content = (content or "").strip().replace("\r\n", "\n")
    if not norm_title and not norm_content:
        return ""
    canonical = f"{norm_title}\n{norm_content}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    if db_path is None:
        return _DEFAULT_DB_PATH
    return Path(db_path)


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize the SQLite semantic cache database and ensure table schema exists."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_query TEXT NOT NULL,
                siis_hash TEXT,
                embedding BLOB NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Backward compatibility: verify column exists for pre-existing tables
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(cache)")
        columns = [row[1] for row in cursor.fetchall()]
        if "siis_hash" not in columns:
            conn.execute("ALTER TABLE cache ADD COLUMN siis_hash TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_siis_hash ON cache(siis_hash)")
        conn.commit()


def clear_cache(db_path: Optional[Union[str, Path]] = None) -> None:
    """Clear all records from the semantic cache."""
    path = _resolve_db_path(db_path)
    if not path.exists():
        return
    with sqlite3.connect(str(path)) as conn:
        conn.execute("DELETE FROM cache")
        conn.commit()


def store(
    query: str,
    embedding: Any = None,
    response: Optional[dict] = None,
    db_path: Optional[Union[str, Path]] = None,
    siis_hash: Optional[str] = None,
) -> None:
    """Store a canonical query, its embedding, and the validated response JSON in SQLite.

    Requirements:
    - Reuses the loaded sentence-transformers model.
    - Scoped by siis_hash to prevent cross-SIIS cache contamination.
    - Embeddings stored as binary BLOB in SQLite.
    - Fails gracefully without raising unexpected exceptions to callers.
    """
    if response is None:
        logger.warning("Attempted to cache None response; skipping store.")
        return

    path = _resolve_db_path(db_path)
    try:
        init_db(path)

        if embedding is None:
            emb_vec = _model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            emb_arr = np.array(emb_vec, dtype=np.float32)
        elif isinstance(embedding, bytes):
            emb_arr = np.frombuffer(embedding, dtype=np.float32)
        else:
            emb_arr = np.array(embedding, dtype=np.float32)

        # Ensure unit normalization for cosine dot-product
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        emb_blob = emb_arr.tobytes()
        response_json = json.dumps(response)
        created_at = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """
                INSERT INTO cache (canonical_query, siis_hash, embedding, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (query.strip(), siis_hash or "", emb_blob, response_json, created_at),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store entry in semantic cache: {e}")


def lookup_with_score(
    query: str,
    threshold: float = 0.85,
    db_path: Optional[Union[str, Path]] = None,
    siis_hash: Optional[str] = None,
) -> Tuple[Optional[dict], float]:
    """Look up a query in semantic cache, returning (best_response_dict, best_cosine_similarity).

    Returns (None, 0.0) if no cached entry reaches the similarity threshold or if the SIIS hash differs.
    """
    path = _resolve_db_path(db_path)
    if not path.exists():
        return None, 0.0

    try:
        init_db(path)
        with sqlite3.connect(str(path)) as conn:
            cursor = conn.cursor()
            if siis_hash:
                cursor.execute(
                    "SELECT id, canonical_query, embedding, response_json FROM cache WHERE siis_hash = ?",
                    (siis_hash,),
                )
            else:
                cursor.execute(
                    "SELECT id, canonical_query, embedding, response_json FROM cache WHERE siis_hash IS NULL OR siis_hash = ''"
                )
            rows = cursor.fetchall()

        if not rows:
            return None, 0.0

        query_vec = _model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        best_score = -1.0
        best_resp_json = None

        for row_id, canonical_query, emb_blob, resp_json in rows:
            try:
                if not isinstance(emb_blob, (bytes, bytearray)):
                    logger.warning(f"Cache row {row_id} has non-binary embedding, skipping")
                    continue

                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if emb.shape[0] != _EMBEDDING_DIM:
                    logger.warning(
                        f"Cache row {row_id} has invalid embedding dimension {emb.shape[0]}, expected {_EMBEDDING_DIM}"
                    )
                    continue

                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                sim = float(np.dot(query_vec, emb))
                if sim > best_score:
                    best_score = sim
                    best_resp_json = resp_json
            except Exception as row_err:
                logger.warning(f"Error evaluating cache row {row_id}: {row_err}")
                continue

        if best_score >= threshold and best_resp_json is not None:
            try:
                parsed_resp = json.loads(best_resp_json)
                return parsed_resp, best_score
            except Exception as parse_err:
                logger.error(f"Corrupted response JSON in cache row: {parse_err}")
                return None, best_score

        return None, max(0.0, best_score)
    except Exception as e:
        logger.error(f"Error during semantic cache lookup: {e}")
        return None, 0.0


def lookup(
    query: str,
    threshold: float = 0.85,
    db_path: Optional[Union[str, Path]] = None,
    siis_hash: Optional[str] = None,
) -> Optional[dict]:
    """Semantic cache lookup returning the highest-similarity cached response when similarity >= threshold.

    Returns None when no cached entry reaches the threshold or if the cache is empty.
    """
    resp, _ = lookup_with_score(
        query, threshold=threshold, db_path=db_path, siis_hash=siis_hash
    )
    return resp

