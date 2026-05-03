"""
Enhanced search module with Jina AI reranking for improved precision.

This module extends the existing search functionality with a two-stage retrieval approach:
1. Initial retrieval: Fast hybrid search (k-NN + keyword with RRF) - gets top 100 candidates
2. Reranking: Precise cross-encoder reranking - refines to top k results

This approach significantly improves precision while maintaining reasonable performance.
"""

import logging
from typing import Any, Dict, List, Optional
from .config import EMBEDDING_FIELD

logger = logging.getLogger(__name__)

# Default RRF parameters
DEFAULT_RRF_RANK_CONSTANT = 60
DEFAULT_RRF_WINDOW_SIZE = 100

# Global reranker cache for performance optimization
_global_reranker_cache = None


def get_cached_reranker() -> Any:
    """Get a cached reranker instance to avoid reloading the model repeatedly."""
    global _global_reranker_cache
    if _global_reranker_cache is None:
        from .embeddings_jina import JinaReranker

        logger.info("Initializing cached Jina reranker...")
        _global_reranker_cache = JinaReranker()
        logger.info("✅ Cached reranker initialized")
    return _global_reranker_cache


def hybrid_search_with_reranking(
    es_client: Any,
    query_vector: List[float],
    user_query: str,
    index_name: str,
    embedding_field: str,
    k: int = 10,
    num_candidates: int = 200,
    rerank_candidates: int = 100,  # How many to retrieve before reranking
    text_query_boost: float = 1.0,
    knn_query_boost: float = 1.0,
    use_reranker: bool = True,
    reranker: Optional[Any] = None,
    rrf_rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    rrf_window_size: int = DEFAULT_RRF_WINDOW_SIZE
) -> List[Dict[str, Any]]:
    """
    Performs hybrid search with optional Jina AI reranking.
    
    Two-stage retrieval process:
    1. Retrieve top `rerank_candidates` using hybrid search (fast)
    2. Rerank using Jina cross-encoder to get top `k` (precise)
    
    Args:
        es_client: Elasticsearch client
        query_vector: Query embedding vector
        user_query: Original text query
        index_name: Elasticsearch index name
        embedding_field: Name of the embedding field
        k: Final number of results to return
        num_candidates: k-NN candidate pool size
        rerank_candidates: Number of candidates to retrieve before reranking
        text_query_boost: Boost for keyword search
        knn_query_boost: Boost for k-NN search
        use_reranker: Whether to use reranking (set False for baseline comparison)
        reranker: JinaReranker instance (will use cached one if None and use_reranker=True)
        rrf_rank_constant: RRF rank constant
        rrf_window_size: RRF window size
    
    Returns:
        List of search hits (reranked if use_reranker=True)
    """
    # Stage 1: Initial retrieval with hybrid search
    # Retrieve more candidates if using reranker
    initial_k = rerank_candidates if use_reranker else k
    
    logger.info(f"Stage 1: Retrieving top {initial_k} candidates using hybrid search...")
    
    search_body = {
        "knn": {
            "field": embedding_field or EMBEDDING_FIELD,
            "query_vector": query_vector,
            "k": initial_k,
            "num_candidates": max(num_candidates, initial_k * 2),
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
                "rank_window_size": max(rrf_window_size, initial_k * 2),
                "rank_constant": rrf_rank_constant
            }
        },
        "_source": [
            "article_id", "content_title", "content_summary", 
            "metadata_products", "content_body", embedding_field or EMBEDDING_FIELD
        ],
        "size": initial_k
    }
    
    try:
        response = es_client.search(
            index=index_name,
            body=search_body,
            request_timeout=90
        )
        hits = response.get('hits', {}).get('hits', [])
        logger.info(f"Retrieved {len(hits)} candidates from Elasticsearch")
        
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}", exc_info=True)
        return []
    
    # Stage 2: Reranking (optional)
    if use_reranker and len(hits) > 0:
        logger.info(f"Stage 2: Reranking top {len(hits)} candidates...")
        
        # Use cached reranker for better performance
        if reranker is None:
            reranker = get_cached_reranker()
        
        try:
            # Prepare documents for reranking
            # Combine title, summary, and beginning of body for reranker
            documents = []
            for hit in hits:
                source = hit.get('_source', {})
                title = source.get('content_title', '')
                summary = source.get('content_summary', '')
                body = source.get('content_body', '')
                
                # Limit body length for reranker (1024 token limit)
                body_preview = body[:500] if body else ''
                
                doc_text = f"{title}\n\n{summary}\n\n{body_preview}"
                documents.append(doc_text)
            
            # Rerank
            ranked_results = reranker.rerank(
                query=user_query,
                documents=documents,
                top_k=k,
                max_length=1024
            )
            
            # Reorder hits based on reranking scores
            reranked_hits = []
            for original_idx, rerank_score in ranked_results:
                hit = hits[original_idx].copy()
                # Add rerank score to hit metadata
                hit['_rerank_score'] = rerank_score
                hit['_original_score'] = hit['_score']
                hit['_score'] = rerank_score  # Replace score with rerank score
                reranked_hits.append(hit)
            
            logger.info(f"Reranking complete. Returning top {len(reranked_hits)} results")
            return reranked_hits
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            logger.warning("Falling back to original search results")
            return hits[:k]
    
    # No reranking - return original results
    return hits[:k]


def knn_search_with_reranking(
    es_client: Any,
    query_vector: List[float],
    user_query: str,
    index_name: str,
    embedding_field: str,
    k: int = 10,
    num_candidates: int = 100,
    rerank_candidates: int = 100,  # Add this parameter
    text_query_boost: float = 1.0,
    knn_query_boost: float = 1.0,
    use_reranker: bool = True,
    reranker: Optional[Any] = None,
    rrf_rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    rrf_window_size: int = DEFAULT_RRF_WINDOW_SIZE
) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper for hybrid_search_with_reranking.
    Drop-in replacement for the original knn_search function.
    
    Set use_reranker=False to disable reranking and get original behavior.
    """
    return hybrid_search_with_reranking(
        es_client=es_client,
        query_vector=query_vector,
        user_query=user_query,
        index_name=index_name,
        embedding_field=embedding_field,
        k=k,
        num_candidates=num_candidates,
        rerank_candidates=rerank_candidates,  # Pass it through
        text_query_boost=text_query_boost,
        knn_query_boost=knn_query_boost,
        use_reranker=use_reranker,
        reranker=reranker,
        rrf_rank_constant=rrf_rank_constant,
        rrf_window_size=rrf_window_size
    )


class SearchPipeline:
    """
    Encapsulates the entire search pipeline with configurable components.
    Useful for A/B testing and comparing different configurations.
    
    Uses cached reranker for optimal performance.
    """
    
    def __init__(
        self,
        es_client: Any,
        index_name: str,
        embedding_field: str = EMBEDDING_FIELD,
        use_reranker: bool = True,
        reranker: Optional[Any] = None
    ):
        """
        Initialize search pipeline.
        
        Args:
            es_client: Elasticsearch client
            index_name: Index to search
            embedding_field: Name of embedding field
            use_reranker: Enable reranking
            reranker: JinaReranker instance (will use cached one if None)
        """
        self.es_client = es_client
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.use_reranker = use_reranker
        
        # Use cached reranker for better performance
        if use_reranker:
            self.reranker = reranker if reranker is not None else get_cached_reranker()
        else:
            self.reranker = None
    
    def search(
        self,
        query_vector: List[float],
        user_query: str,
        k: int = 10,
        num_candidates: int = 200,
        text_query_boost: float = 1.0,
        knn_query_boost: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Execute search with configured pipeline.
        
        Returns:
            List of search hits
        """
        return knn_search_with_reranking(
            es_client=self.es_client,
            query_vector=query_vector,
            user_query=user_query,
            index_name=self.index_name,
            embedding_field=self.embedding_field,
            k=k,
            num_candidates=num_candidates,
            text_query_boost=text_query_boost,
            knn_query_boost=knn_query_boost,
            use_reranker=self.use_reranker,
            reranker=self.reranker
        )
    
    def compare_with_baseline(
        self,
        query_vector: List[float],
        user_query: str,
        k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare reranked results with baseline (no reranking).
        Useful for evaluation and A/B testing.
        
        Returns:
            Dictionary with 'baseline' and 'reranked' results
        """
        # Get baseline results (no reranking)
        baseline_hits = knn_search_with_reranking(
            es_client=self.es_client,
            query_vector=query_vector,
            user_query=user_query,
            index_name=self.index_name,
            embedding_field=self.embedding_field,
            k=k,
            use_reranker=False
        )
        
        # Get reranked results (use cached reranker)
        reranked_hits = knn_search_with_reranking(
            es_client=self.es_client,
            query_vector=query_vector,
            user_query=user_query,
            index_name=self.index_name,
            embedding_field=self.embedding_field,
            k=k,
            use_reranker=True,
            reranker=self.reranker
        )
        
        return {
            'baseline': baseline_hits,
            'reranked': reranked_hits
        }


# Example usage
if __name__ == "__main__":
    print("=== Jina AI Search with Optimized Reranking ===\n")
    
    print("🚀 Performance Optimizations:")
    print("✅ Cached reranker - no model reloading between queries")
    print("✅ Streamlit caching for models (@st.cache_resource)")
    print("✅ Configurable rerank candidate count")
    print("✅ Fallback to baseline if reranking fails")
    
    print("\n1. Basic usage (optimized performance):")
    print("""
    from src.es_client import get_es_client
    from src.search_jina import knn_search_with_reranking
    from src.embeddings_jina import load_jina_embedding_model
    from src.config import INDEX_NAME, EMBEDDING_FIELD
    
    # Initialize (models are cached in Streamlit)
    es_client = get_es_client()
    model = load_jina_embedding_model()
    
    # Search with optimized reranking
    user_query = "How to configure Elasticsearch cluster?"
    query_vector = model.encode(user_query, task='retrieval.query')
    
    hits = knn_search_with_reranking(
        es_client=es_client,
        query_vector=query_vector,
        user_query=user_query,
        index_name=INDEX_NAME,
        embedding_field=EMBEDDING_FIELD,
        k=10,
        use_reranker=True  # Uses cached reranker automatically
    )
    """)
    
    print("\n2. Performance tuning:")
    print("""
    # Reduce rerank candidates for faster performance
    hits = knn_search_with_reranking(
        # ... other params ...
        rerank_candidates=50,  # Instead of default 100
        use_reranker=True
    )
    
    # Or disable reranking for speed-critical applications
    hits = knn_search_with_reranking(
        # ... other params ...
        use_reranker=False  # Fast baseline search only
    )
    """)
    
    print("\n=== Expected Performance Improvements ===")
    print("🚀 First query: ~30-60s (model loading)")
    print("⚡ Subsequent queries: ~2-5s (cached models)")
    print("📊 Baseline search: ~0.5-1s (no reranking)")
    
    print("\nRun 'streamlit run run_pipeline.py' to test the optimized UI!")
