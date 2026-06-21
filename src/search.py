import logging
from typing import Any, Dict, List
from .config import EMBEDDING_FIELD

# Configure logging
logger = logging.getLogger(__name__)

# Default RRF parameters (can be overridden)
DEFAULT_RRF_RANK_CONSTANT = 60
DEFAULT_RRF_WINDOW_SIZE = 100

def knn_search(
    es_client: Any,
    query_vector: List[float],
    user_query: str,
    index_name: str,
    embedding_field: str, # Keep this argument for clarity, though we use the imported constant
    k: int = 10,
    num_candidates: int = 100,
    text_query_boost: float = 1.0,
    knn_query_boost: float = 1.0,
    rrf_rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    rrf_window_size: int = DEFAULT_RRF_WINDOW_SIZE
) -> List[Dict[str, Any]]:
    """
    Performs a hybrid k-NN and text search using Reciprocal Rank Fusion (RRF).
    Requests source fields and explicitly requests the embedding field via 'fields'.
    """
    search_body = {
        "knn": {
            "field": embedding_field or EMBEDDING_FIELD,
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates,
            "boost": knn_query_boost
        },
        "query": {
            "multi_match": {
                "query": user_query,
                "fields": ["content_title^3", "content_summary^2", "content_body"],
                "fuzziness": "AUTO",
                "boost": text_query_boost
            }
        },
        "rank": {
            "rrf": {
                "rank_window_size": max(rrf_window_size, k * 2),
                "rank_constant": rrf_rank_constant
            }
        },
        # Request embedding in _source again
        "_source": ["article_id", "content_title", "content_summary", "metadata_products", "content_body", embedding_field or EMBEDDING_FIELD],
        # Remove 'fields' parameter
        "size": k
    }

    try:
        logger.debug(f"Elasticsearch search body: {search_body}")
        response = es_client.search(
            index=index_name,
            body=search_body,
            request_timeout=90
        )
        return response.get('hits', {}).get('hits', [])
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}", exc_info=True)
        return []

def get_article_by_id(
    es_client: Any,
    article_id: str,
    index_name: str,
    embedding_field: str = EMBEDDING_FIELD,
    include_embedding: bool = False,
) -> Dict[str, Any]:
    """
    Fetch a single KB article (chunk) by its ``article_id`` term.

    Returns the raw Elasticsearch hit (``{"_id", "_score", "_source", ...}``) for
    the first match, or ``None`` if no document with that ``article_id`` exists.
    Used by both interactive look-ups and the read-only MCP ``get_chunk`` tool, so
    the fetch logic lives in one place. Errors propagate to the caller (the MCP
    layer classifies transient backend errors via its ``guard``).
    """
    source_fields = ["article_id", "content_title", "content_summary", "metadata_products", "content_body"]
    if include_embedding:
        source_fields.append(embedding_field or EMBEDDING_FIELD)

    search_body = {
        "query": {"term": {"article_id": article_id}},
        "_source": source_fields,
        "size": 1,
    }

    response = es_client.search(index=index_name, body=search_body, request_timeout=30)
    hits = response.get("hits", {}).get("hits", [])
    return hits[0] if hits else None


def run_search_and_visualize(es_client, query_vector, user_query, index_name, k, num_candidates):
    print("Running hybrid k-NN and text search...")
    hits = knn_search(
        es_client=es_client,
        query_vector=query_vector,
        user_query=user_query,
        index_name=index_name,
        embedding_field=EMBEDDING_FIELD,
        k=k,
        num_candidates=num_candidates
    )

    print(f"knn_search returned {len(hits)} hits.")
    if not hits:
        print("No hits found by hybrid search. Skipping analysis and visualization.")
        return # Exit the function if no hits

    print("Analyzing hits...")
    df_hits = analyze_knn_hits(hits)
    print(df_hits.head())
    visualize_hits(df_hits, x_col="score", y_col="rank")
