"""FastMCP server for the duplicate-detection core.

Exposes the existing KB duplicate-detection pipeline as three READ-ONLY MCP tools
that any MCP client (Claude Code, Cursor, a custom agent) can call:

- ``hybrid_search`` — kNN + keyword (RRF-fused) retrieval over the KB index.
- ``find_duplicates`` — near-duplicate articles for a seed text or chunk_id.
- ``get_chunk`` — fetch a single article by its chunk_id.

The tool *logic* lives in :mod:`src.mcp.tools` as plain functions; the wrappers
here are thin: they supply the cached resource singletons (:mod:`src.mcp.
resources`) and forward to the impls. There is NO ingestion / write tool, and the
server performs no mutations. ``MCP_ALLOW_MUTATIONS`` defaults to false and is
reserved for future use; the current toolset is read-only regardless.

Transport is selected by ``MCP_TRANSPORT``: ``stdio`` (default, for local dev and
Claude Code) or ``http`` (streamable-HTTP). Run it from the repository root::

    python -m src.mcp.server
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .resources import (
    get_embedder,
    get_embedding_field,
    get_es_client,
    get_index_name,
    get_kb_base_url,
)
from .tools import find_duplicates_impl, get_chunk_impl, hybrid_search_impl

mcp = FastMCP("duplicate-detection")


@mcp.tool()
def hybrid_search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Hybrid (semantic + keyword) search over the knowledge-base articles. Read-only.

    WHAT IT DOES: Embeds the query with the Jina v3 model, then runs the existing
    two-stage hybrid retrieval — dense-vector kNN AND a `multi_match` keyword query
    fused with Elasticsearch Reciprocal Rank Fusion (RRF) — over the KB index. It
    returns the top-k matching articles with provenance (article_id, scores).

    WHEN TO USE: To find KB articles relevant to a topic or natural-language query,
    or to gather candidate articles before deciding what to read or compare.

    WHEN NOT TO USE: To find near-duplicates of a specific article, use
    `find_duplicates`. To fetch one known article in full, use `get_chunk`.

    INPUTS:
      - query (str, required): natural-language search query (non-empty).
      - filters (object, optional): simple case-insensitive substring filters on
        returned fields, e.g. {"title": "kibana"} or {"products": "Elasticsearch"}.
        Applied after retrieval to the projected result shape.
      - k (int, default 10, 1..100): number of results to return.

    OUTPUT: {query, reranked (bool), count, results: [{chunk_id, article_id, title,
    summary, body_preview, products, score, url}]}. Results are ordered best-first
    and may be empty. The raw embedding vector and raw ES response are never
    returned.

    EDGE CASES & FAILURES: An empty `results` list with no error means nothing
    matched. On failure a structured error is returned instead: errorCategory=
    "validation" (empty query, k out of range — not retryable), "transient"
    (search backend unreachable — retryable). Stack traces are never returned.
    """
    return hybrid_search_impl(
        query,
        filters=filters,
        k=k,
        es_client=get_es_client(),
        embedder=get_embedder(),
        index_name=get_index_name(),
        embedding_field=get_embedding_field(),
        kb_base_url=get_kb_base_url(),
    )


@mcp.tool()
def find_duplicates(
    text: Optional[str] = None,
    chunk_id: Optional[str] = None,
    threshold: float = 0.9,
) -> Dict[str, Any]:
    """Find near-duplicate KB articles for a seed text or article. Read-only.

    WHAT IT DOES: Resolves a seed (raw `text`, or the title+summary of an existing
    article when `chunk_id` is given), gathers candidate articles via the hybrid
    search, then applies the project's dependency-free pairwise string-similarity
    helper (`detect_duplicates`) at the requested `threshold`. Returns the
    candidates whose normalized similarity to the seed meets the threshold.

    WHEN TO USE: To check whether an article (by chunk_id) or a draft (by text)
    duplicates existing KB content, or to surface merge/near-duplicate candidates.

    WHEN NOT TO USE: For general topical search use `hybrid_search`. To read one
    article in full use `get_chunk`.

    INPUTS:
      - text (str, optional): seed text to check for duplicates.
      - chunk_id (str, optional): article_id of an existing article to use as the
        seed. Provide EXACTLY ONE of `text` or `chunk_id`.
      - threshold (float, default 0.9, 0.0..1.0): minimum normalized string
        similarity (0=anything, 1=identical) for a candidate to count as a
        duplicate. 0.9 is the project's tuned default; lower it to widen recall.

    OUTPUT: {seed (the seed article shape, or null when seeded by raw text),
    threshold, count, duplicates: [{score, candidate: {chunk_id, title, summary,
    ...}}]}, ordered by descending similarity.

    EDGE CASES & FAILURES: An empty `duplicates` list with no error means no
    candidate met the threshold. Structured errors: "validation" (neither/both of
    text+chunk_id, threshold out of range — not retryable), "business" (unknown
    chunk_id, or an empty seed — not retryable), "transient" (backend unreachable —
    retryable). Stack traces are never returned.
    """
    return find_duplicates_impl(
        text,
        chunk_id,
        threshold=threshold,
        es_client=get_es_client(),
        embedder=get_embedder(),
        index_name=get_index_name(),
        embedding_field=get_embedding_field(),
        kb_base_url=get_kb_base_url(),
    )


@mcp.tool()
def get_chunk(chunk_id: str) -> Dict[str, Any]:
    """Fetch a single KB article by its chunk_id (article_id). Read-only.

    WHAT IT DOES: Looks up the article whose `article_id` equals `chunk_id` and
    returns its full projected shape, including the complete body. No embedding
    vector or raw ES response is returned.

    WHEN TO USE: When you already have a chunk_id (e.g. from `hybrid_search` or
    `find_duplicates`) and want to read that article's full content.

    WHEN NOT TO USE: To search by topic use `hybrid_search`; to find similar
    articles use `find_duplicates`.

    INPUTS:
      - chunk_id (str, required): the article_id to fetch (non-empty).

    OUTPUT: {chunk: {chunk_id, article_id, title, summary, body, products, url}}.

    EDGE CASES & FAILURES: Structured errors: "validation" (empty chunk_id — not
    retryable), "business" (no article with that id — not retryable), "transient"
    (backend unreachable — retryable). Stack traces are never returned.
    """
    return get_chunk_impl(
        chunk_id,
        es_client=get_es_client(),
        index_name=get_index_name(),
        kb_base_url=get_kb_base_url(),
    )


def main() -> None:
    """Run the FastMCP server on the configured transport (``MCP_TRANSPORT``)."""
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
