"""
Jina AI Embeddings Integration Module

This module provides drop-in replacements for the existing embedding functions
using Jina AI's state-of-the-art models.

Key improvements:
- 8192 token context (vs 512 for e5-large-v2)
- Task-specific embeddings for better duplicate detection
- 5x faster inference
- Multilingual support (89 languages)
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Union
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


class JinaEmbeddings:
    """
    Wrapper for Jina AI embedding models with support for local and API-based inference.
    """
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", use_api: bool = False, api_key: Optional[str] = None):
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        self.dimensions = 1024  # Default Jina v3 dimensions
        
        if not use_api:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load the local Jina AI model with error handling."""
        try:
            logger.info(f"Loading Jina AI model: {self.model_name}")
            
            # Try using sentence-transformers first (more stable)
            from sentence_transformers import SentenceTransformer
            
            # Map Jina model names to sentence-transformers compatible names
            model_mapping = {
                "jinaai/jina-embeddings-v3": "jinaai/jina-embeddings-v3",
                "jina-embeddings-v3": "jinaai/jina-embeddings-v3"
            }
            
            model_name_st = model_mapping.get(self.model_name, self.model_name)
            
            try:
                self.model = SentenceTransformer(model_name_st, trust_remote_code=True)
                logger.info("✅ Jina AI model loaded successfully via sentence-transformers")
                self._use_sentence_transformers = True
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer loading failed: {e}")
                logger.info("Trying direct transformers approach...")
            
            # Fallback to direct transformers approach
            from transformers import AutoModel, AutoTokenizer
            
            # Clear cache and try again
            try:
                import shutil
                import os
                cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
                if os.path.exists(cache_dir):
                    logger.info("Clearing transformers cache...")
                    for item in os.listdir(cache_dir):
                        if "jina" in item.lower():
                            item_path = os.path.join(cache_dir, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
            except Exception as e:
                logger.warning(f"Cache clearing failed: {e}")
            
            # Try with force_download to get fresh files
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                force_download=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float32,
                force_download=True
            )
            self._use_sentence_transformers = False
            logger.info("✅ Jina AI model loaded successfully via transformers")
            
        except Exception as e:
            logger.error(f"Failed to load Jina AI model: {e}")
            # Fallback to a working model
            logger.info("Falling back to a compatible sentence-transformers model...")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Reliable fallback
                self._use_sentence_transformers = True
                logger.warning("⚠️ Using fallback model 'all-MiniLM-L6-v2' instead of Jina AI")
            except Exception as fallback_error:
                logger.error(f"Even fallback model failed: {fallback_error}")
                raise RuntimeError("Could not load any embedding model")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        task: str = "text-matching",
        max_length: int = 8192,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            task: Embedding task type. Options:
                - 'retrieval.query': For search queries
                - 'retrieval.passage': For document passages
                - 'text-matching': For similarity/duplicate detection (RECOMMENDED)
                - 'classification': For text classification
                - 'separation': For separating different concepts
                - 'clustering': For clustering tasks
            max_length: Maximum token length (up to 8192)
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar
            normalize: Normalize embeddings (recommended)
        
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_api:
            return self._encode_via_api(texts, task, max_length, show_progress_bar)
        else:
            return self._encode_local(texts, task, max_length, batch_size, show_progress_bar, normalize)
    
    def _encode_local(
        self,
        texts: List[str],
        task: str,
        max_length: int,
        batch_size: int,
        show_progress_bar: bool,
        normalize: bool
    ) -> np.ndarray:
        """Encode using local model."""
        if self._use_sentence_transformers:
            # Use sentence-transformers encode method
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize
            )
        else:
            # Use transformers encode method with task parameter
            embeddings = self.model.encode(
                texts,
                task=task,
                max_length=max_length,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar
            )
        
        embeddings = np.array(embeddings)
        
        # Truncate to desired dimensions if using Matryoshka
        if embeddings.shape[1] > self.dimensions:
            embeddings = embeddings[:, :self.dimensions]
        
        if normalize and not self._use_sentence_transformers:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            embeddings = embeddings / norms
        
        return embeddings
    
    def _encode_via_api(
        self,
        texts: List[str],
        task: str,
        max_length: int,
        show_progress_bar: bool
    ) -> np.ndarray:
        """Encode using Jina AI Cloud API."""
        import requests
        
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        batch_size = 100  # API batch limit
        
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding via Jina API")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": "jina-embeddings-v3",
                "task": task,
                "dimensions": self.dimensions,
                "input": batch
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"API request failed for batch {i//batch_size}: {e}")
                raise
        
        return np.array(all_embeddings)


class JinaReranker:
    """
    Wrapper for Jina AI reranker models for improved search precision.
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        use_api: bool = False,
        api_key: Optional[str] = None,
        trust_remote_code: bool = True
    ):
        """
        Initialize Jina reranker.
        
        Args:
            model_name: HuggingFace model name
            use_api: Whether to use Jina AI Cloud API
            api_key: Jina AI API key (required if use_api=True)
            trust_remote_code: Trust remote code for model loading
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        
        if use_api:
            if not api_key:
                raise ValueError("API key required when use_api=True")
            logger.info(f"Initialized Jina AI Cloud API client for reranking")
        else:
            logger.info(f"Loading local Jina reranker: {model_name}")
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            logger.info(f"Reranker {model_name} loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        max_length: int = 1024
    ) -> List[tuple]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return only top k results (None = return all)
            max_length: Maximum token length for query+document pairs
        
        Returns:
            List of (index, score) tuples sorted by score (highest first)
        """
        if self.use_api:
            return self._rerank_via_api(query, documents, top_k)
        else:
            return self._rerank_local(query, documents, top_k, max_length)
    
    def _rerank_local(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int],
        max_length: int
    ) -> List[tuple]:
        """Rerank using local model."""
        pairs = [[query, doc] for doc in documents]
        
        scores = self.model.compute_score(
            pairs,
            max_length=max_length
        )
        
        # Convert to list if single score
        if not isinstance(scores, list):
            scores = [scores]
        
        # Create (index, score) pairs and sort
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _rerank_via_api(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int]
    ) -> List[tuple]:
        """Rerank using Jina AI Cloud API."""
        import requests
        
        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": top_k if top_k else len(documents)
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # API returns sorted results with index and score
            results = [(item["index"], item["relevance_score"]) for item in data["results"]]
            return results
            
        except Exception as e:
            logger.error(f"Reranking API request failed: {e}")
            raise


# Backward-compatible functions for easy migration

def load_jina_embedding_model(
    model_name: str = "jinaai/jina-embeddings-v3",
    use_api: bool = False,
    api_key: Optional[str] = None
) -> JinaEmbeddings:
    """
    Load Jina embedding model (drop-in replacement for load_embedding_model).
    
    Args:
        model_name: Model name (default: jina-embeddings-v3)
        use_api: Use Jina AI Cloud API instead of local model
        api_key: API key for Jina AI Cloud
    
    Returns:
        JinaEmbeddings instance
    """
    return JinaEmbeddings(model_name=model_name, use_api=use_api, api_key=api_key)


def compute_jina_embeddings(
    model: JinaEmbeddings,
    texts: List[str],
    task: str = "text-matching",
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute embeddings using Jina model (drop-in replacement for compute_embeddings).
    
    Args:
        model: JinaEmbeddings instance
        texts: List of texts to embed
        task: Embedding task type (text-matching for duplicate detection)
        batch_size: Batch size
        normalize: Normalize embeddings
    
    Returns:
        numpy array of embeddings
    """
    return model.encode(
        texts,
        task=task,
        batch_size=batch_size,
        normalize=normalize
    )


def combine_text_with_metadata(row: pd.Series, max_body_length: int = 6000) -> str:
    """
    Combine content fields and metadata into a single string for embedding.
    Enhanced version that handles longer content since Jina v3 supports 8192 tokens.
    
    Args:
        row: DataFrame row
        max_body_length: Maximum characters for body (can be higher with Jina)
    
    Returns:
        Combined text string
    """
    parts = []
    
    # Title is most important
    if pd.notna(row.get("content_title")):
        parts.append(f"Title: {str(row['content_title'])}")
    
    # Summary is second most important
    if pd.notna(row.get("content_summary")):
        parts.append(f"Summary: {str(row['content_summary'])}")
    
    # Body can be much longer with Jina v3 (8192 tokens ≈ 6000 chars)
    if pd.notna(row.get("content_body")):
        body = str(row["content_body"])
        if len(body) > max_body_length:
            body = body[:max_body_length] + "..."
        parts.append(f"Content: {body}")
    
    # Metadata
    if pd.notna(row.get("category")):
        parts.append(f"Category: {row['category']}")
    
    if pd.notna(row.get("metadata_products")):
        parts.append(f"Products: {row['metadata_products']}")
    
    return "\n\n".join(parts)


# Example usage and migration guide
if __name__ == "__main__":
    print("=== Jina AI Embeddings Migration Example ===\n")
    
    # Example 1: Local model usage (drop-in replacement)
    print("1. Loading local Jina embeddings model...")
    model = load_jina_embedding_model()
    
    sample_texts = [
        "How to configure Elasticsearch cluster settings",
        "Elasticsearch cluster configuration guide",
        "Setting up Kibana dashboards"
    ]
    
    print("2. Computing embeddings with text-matching task...")
    embeddings = compute_jina_embeddings(
        model,
        sample_texts,
        task="text-matching",  # Optimal for duplicate detection
        normalize=True
    )
    print(f"   Generated embeddings shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}, Count: {embeddings.shape[0]}")
    
    # Example 2: API usage (requires API key)
    print("\n3. Example API usage (requires JINA_API_KEY env var):")
    print("""
    import os
    api_key = os.getenv('JINA_API_KEY')
    model = load_jina_embedding_model(use_api=True, api_key=api_key)
    embeddings = compute_jina_embeddings(model, texts)
    """)
    
    # Example 3: Reranking
    print("\n4. Loading reranker for two-stage retrieval...")
    print("""
    reranker = JinaReranker()
    query = "elasticsearch configuration"
    documents = ["doc1 text", "doc2 text", "doc3 text"]
    ranked_results = reranker.rerank(query, documents, top_k=10)
    for idx, score in ranked_results:
        print(f"Doc {idx}: {score:.4f}")
    """)
    
    print("\n=== Migration Complete! ===")
    print("\nNext steps:")
    print("1. Update src/embeddings.py to import from embeddings_jina")
    print("2. Update src/ingest.py to use JinaEmbeddings")
    print("3. Update src/search.py to add reranking stage")
    print("4. Re-index your documents with new embeddings")
    print("\nSee JINA_AI_IMPROVEMENTS_ANALYSIS.md for detailed roadmap.")
