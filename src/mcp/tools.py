"""MCP tool handlers wrapping the duplicate-detection retrieval / similarity core.

These are plain, importable functions — no FastMCP or HTTP coupling — so they can
be unit-tested directly with a fake Elasticsearch client and a fake embedder.
``src/mcp/server.py`` registers thin FastMCP wrappers that supply the cached
resource singletons (:mod:`src.mcp.resources`).

Each handler is THIN: it validates inputs with pydantic, calls an EXISTING ``src``
function (``hybrid_search_with_reranking``, ``detect_duplicates``,
``get_article_by_id``), and returns a small JSON-serialisable shape. It never
returns a raw Elasticsearch response, and never embeds business logic of its own.

Every handler is wrapped by :func:`src.mcp.errors.guard`, so it either returns a
structured success payload or a structured error payload — never a raised
exception or a stack trace. All three tools are READ-ONLY.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..deduplication import detect_duplicates
from ..search import get_article_by_id
from ..search_jina import hybrid_search_with_reranking
from .errors import ToolBusinessError, ToolValidationError, guard

MAX_K = 100
DEFAULT_K = 10
# How many candidates to gather before running pairwise similarity in find_duplicates.
DEFAULT_DUP_CANDIDATES = 25


# --------------------------------------------------------------------------- #
# Input validation models (pydantic). Kept tiny and local to each tool.
# --------------------------------------------------------------------------- #
class _HybridSearchInput(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)
    filters: Optional[Dict[str, Any]] = None


class _FindDuplicatesInput(BaseModel):
    text: Optional[str] = None
    chunk_id: Optional[str] = None
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    k: int = Field(default=DEFAULT_DUP_CANDIDATES, ge=1, le=MAX_K)


class _GetChunkInput(BaseModel):
    chunk_id: str = Field(min_length=1)


def _raise_validation(exc: ValidationError) -> None:
    """Translate a pydantic ValidationError into a structured ToolValidationError."""
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", ())) or "input"
    raise ToolValidationError(f"Invalid `{loc}`: {first.get('msg', 'invalid value')}.")


def _embed_query(embedder: Any, query: str) -> List[float]:
    """Embed a query string into a list-of-floats vector.

    Mirrors the Streamlit path (`model.encode([query], task='retrieval.query')[0]`)
    and normalises the result to a plain Python list regardless of whether the
    embedder returns a numpy array or a list.
    """
    vector = embedder.encode([query], task="retrieval.query", show_progress_bar=False)[0]
    tolist = getattr(vector, "tolist", None)
    return tolist() if callable(tolist) else list(vector)


def _hit_to_shape(hit: Dict[str, Any], *, kb_base_url: str = "") -> Dict[str, Any]:
    """Project a raw ES hit into a small, citation-friendly, JSON-safe shape.

    Preserves provenance (article_id, scores) and never leaks the embedding vector
    or the full raw ES response.
    """
    source = hit.get("_source", {}) or {}
    article_id = source.get("article_id")
    body = source.get("content_body") or ""
    shape: Dict[str, Any] = {
        "chunk_id": article_id,
        "article_id": article_id,
        "title": source.get("content_title", ""),
        "summary": source.get("content_summary", ""),
        "body_preview": body[:500],
        "products": source.get("metadata_products"),
        "score": hit.get("_score"),
        "url": f"{kb_base_url}{article_id}" if (kb_base_url and article_id) else None,
    }
    # Reranking provenance, when present — never drop these (CLAUDE.md invariant 4).
    if "_rerank_score" in hit:
        shape["rerank_score"] = hit["_rerank_score"]
    if "_original_score" in hit:
        shape["original_score"] = hit["_original_score"]
    return shape


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
@guard("hybrid_search")
def hybrid_search_impl(
    query: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    k: int = DEFAULT_K,
    es_client: Any,
    embedder: Any,
    index_name: str,
    embedding_field: str,
    kb_base_url: str = "",
    use_reranker: bool = False,
) -> Dict[str, Any]:
    """Hybrid (kNN + keyword, RRF-fused) search over the KB index. Read-only.

    Thin wrapper over ``src.search_jina.hybrid_search_with_reranking``: embeds the
    query, runs the existing two-stage retrieval, and returns filtered hit shapes.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolValidationError("`query` must be a non-empty string.")
    try:
        params = _HybridSearchInput(query=query.strip(), k=k, filters=filters)
    except ValidationError as exc:
        _raise_validation(exc)

    query_vector = _embed_query(embedder, params.query)

    hits = hybrid_search_with_reranking(
        es_client=es_client,
        query_vector=query_vector,
        user_query=params.query,
        index_name=index_name,
        embedding_field=embedding_field,
        k=params.k,
        use_reranker=use_reranker,
    )

    results = [_hit_to_shape(hit, kb_base_url=kb_base_url) for hit in hits]

    # Optional, simple post-filter on returned _source fields (e.g. product area).
    # Kept here because it operates on the projected shape, not on ES internals.
    if params.filters:
        results = [r for r in results if _matches_filters(r, params.filters)]

    return {
        "query": params.query,
        "reranked": bool(use_reranker),
        "count": len(results),
        "results": results,
    }


def _matches_filters(result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Case-insensitive substring match of each filter value against the named field."""
    for key, value in filters.items():
        field_value = result.get(key)
        if field_value is None:
            return False
        if str(value).casefold() not in str(field_value).casefold():
            return False
    return True


@guard("find_duplicates")
def find_duplicates_impl(
    text: Optional[str] = None,
    chunk_id: Optional[str] = None,
    *,
    threshold: float = 0.9,
    k: int = DEFAULT_DUP_CANDIDATES,
    es_client: Any,
    embedder: Any,
    index_name: str,
    embedding_field: str,
    kb_base_url: str = "",
) -> Dict[str, Any]:
    """Find near-duplicate KB articles for a seed (raw `text` or an existing
    `chunk_id`). Read-only.

    Resolves the seed text (fetching the article when a `chunk_id` is given),
    gathers candidate articles via the existing hybrid search, then applies the
    existing dependency-free ``src.deduplication.detect_duplicates`` string
    similarity helper at the requested `threshold`. Only candidate/seed pairs are
    reported, so the seed is always index 0.
    """
    try:
        params = _FindDuplicatesInput(text=text, chunk_id=chunk_id, threshold=threshold, k=k)
    except ValidationError as exc:
        _raise_validation(exc)

    if not params.text and not params.chunk_id:
        raise ToolValidationError("Provide either `text` or `chunk_id`.")
    if params.text and params.chunk_id:
        raise ToolValidationError("Provide only one of `text` or `chunk_id`, not both.")

    # Resolve the seed text.
    seed_meta: Optional[Dict[str, Any]] = None
    if params.chunk_id:
        hit = get_article_by_id(es_client, params.chunk_id, index_name)
        if hit is None:
            raise ToolBusinessError(
                f"No article found with chunk_id '{params.chunk_id}'.",
                details={"chunk_id": params.chunk_id},
            )
        seed_meta = _hit_to_shape(hit, kb_base_url=kb_base_url)
        source = hit.get("_source", {}) or {}
        seed_text = " ".join(
            part for part in (source.get("content_title"), source.get("content_summary")) if part
        ).strip()
        if not seed_text:
            seed_text = (source.get("content_body") or "").strip()
    else:
        seed_text = params.text.strip()

    if not seed_text:
        raise ToolBusinessError("The seed text is empty after normalisation; nothing to compare.")

    # Gather candidate articles via the existing hybrid search (no reranker needed
    # for a similarity screen).
    query_vector = _embed_query(embedder, seed_text)
    hits = hybrid_search_with_reranking(
        es_client=es_client,
        query_vector=query_vector,
        user_query=seed_text,
        index_name=index_name,
        embedding_field=embedding_field,
        k=params.k,
        use_reranker=False,
    )

    candidates = [_hit_to_shape(hit, kb_base_url=kb_base_url) for hit in hits]
    # Drop the seed itself from the candidate list when searching by chunk_id.
    if params.chunk_id:
        candidates = [c for c in candidates if c.get("chunk_id") != params.chunk_id]

    if not candidates:
        return {"seed": seed_meta, "threshold": params.threshold, "count": 0, "duplicates": []}

    # Run the EXISTING similarity helper. Index 0 is the seed; 1..N are candidates.
    candidate_texts = [
        " ".join(p for p in (c.get("title"), c.get("summary")) if p).strip() or c.get("body_preview", "")
        for c in candidates
    ]
    pairs = detect_duplicates([seed_text, *candidate_texts], threshold=params.threshold)

    duplicates = []
    for pair in pairs:
        # Only keep pairs that involve the seed (index 0).
        if pair["left_index"] != 0:
            continue
        candidate = candidates[pair["right_index"] - 1]
        duplicates.append({"score": pair["score"], "candidate": candidate})

    duplicates.sort(key=lambda d: d["score"], reverse=True)
    return {
        "seed": seed_meta,
        "threshold": params.threshold,
        "count": len(duplicates),
        "duplicates": duplicates,
    }


@guard("get_chunk")
def get_chunk_impl(
    chunk_id: str,
    *,
    es_client: Any,
    index_name: str,
    kb_base_url: str = "",
) -> Dict[str, Any]:
    """Fetch a single KB article (chunk) by its `chunk_id` (article_id). Read-only.

    Thin wrapper over ``src.search.get_article_by_id``. Returns the projected,
    JSON-safe article shape (including the full body), or a structured business
    error when no such article exists.
    """
    try:
        params = _GetChunkInput(chunk_id=chunk_id)
    except ValidationError as exc:
        _raise_validation(exc)

    hit = get_article_by_id(es_client, params.chunk_id, index_name)
    if hit is None:
        raise ToolBusinessError(
            f"No article found with chunk_id '{params.chunk_id}'.",
            details={"chunk_id": params.chunk_id},
        )

    shape = _hit_to_shape(hit, kb_base_url=kb_base_url)
    # get_chunk returns the full body, not just a preview.
    source = hit.get("_source", {}) or {}
    shape["body"] = source.get("content_body") or ""
    shape.pop("body_preview", None)
    shape.pop("score", None)
    return {"chunk": shape}
