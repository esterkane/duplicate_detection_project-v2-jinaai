import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Import Jina AI functions
from .embeddings_jina import load_jina_embedding_model, compute_jina_embeddings
from .config import ES_MODEL_ID, JINA_MODEL_NAME, JINA_TASK, JINA_MAX_LENGTH

logger = logging.getLogger(__name__)

# Global model instance for reuse
_jina_model = None

def get_jina_embedding_model():
    """Get or load the Jina AI embedding model."""
    global _jina_model
    if _jina_model is None:
        logger.info("Loading Jina AI embedding model...")
        _jina_model = load_jina_embedding_model(
            model_name=JINA_MODEL_NAME,
            use_api=False  # Using local model
        )
        logger.info("Jina AI model loaded successfully")
    return _jina_model

def get_available_models(es_client: Elasticsearch) -> list:
    """
    Retrieve available trained text embedding model IDs from Elasticsearch.
    """
    models = es_client.ml.get_trained_models()
    return [m["model_id"] for m in models.get("trained_model_configs", [])]


def select_model(es_client: Elasticsearch,
                 fallback_model: str = "multilingual-e5-small") -> str:
    """
    Select the first available model, or fall back to a default.
    """
    model_ids = get_available_models(es_client)
    return model_ids[0] if model_ids else fallback_model


def combine_text_with_metadata(row: pd.Series) -> str:
    """
    Combine content fields and metadata into a single string for embedding.
    """
    parts = []
    if pd.notna(row.get("content_title")):
        parts.append(str(row["content_title"]))
    if pd.notna(row.get("content_summary")):
        parts.append(str(row["content_summary"]))
    if pd.notna(row.get("content_body")):
        parts.append(str(row["content_body"]))
    if pd.notna(row.get("category")):
        parts.append(f"Category: {row['category']}")
    if pd.notna(row.get("metadata_products")):
        parts.append(f"Products: {row['metadata_products']}")
    return "\n".join(parts)


def get_embedding(text: str, es_client: Elasticsearch = None, model_id: str = None) -> np.ndarray:
    """
    Generate embeddings using Jina AI (preferred) or fallback to Elasticsearch ML.
    
    Args:
        text: Text to embed
        es_client: Elasticsearch client (for fallback)
        model_id: Model ID (for fallback)
    
    Returns:
        Embedding vector as numpy array
    """
    try:
        # Use Jina AI embeddings
        model = get_jina_embedding_model()
        embeddings = compute_jina_embeddings(
            model=model,
            texts=[text],
            task=JINA_TASK,
            batch_size=1,
            normalize=True
        )
        return embeddings[0]  # Return single embedding
        
    except Exception as e:
        logger.error(f"Jina AI embedding failed: {e}")
        # Fallback to original Elasticsearch ML method
        if es_client and model_id:
            logger.info("Falling back to Elasticsearch ML embeddings")
            return get_embedding_from_elasticsearch(text, es_client, model_id)
        else:
            raise e

def get_embedding_from_elasticsearch(text: str, es_client: Elasticsearch, model_id: str) -> np.ndarray:
    """
    Original Elasticsearch ML embedding method (fallback).
    """
    try:
        response = es_client.ml.infer_trained_model(
            model_id=model_id,
            body={"docs": [{"text_field": text}]}
        )
        embedding = response['inference_results'][0]['predicted_value']
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to get embedding from Elasticsearch: {e}")
        raise

def get_jina_embeddings_batch(texts: list, task: str = None) -> np.ndarray:
    """
    Generate embeddings for multiple texts using Jina AI.
    
    Args:
        texts: List of texts to embed
        task: Jina AI task type (defaults to config value)
    
    Returns:
        Array of embeddings
    """
    model = get_jina_embedding_model()
    embeddings = compute_jina_embeddings(
        model=model,
        texts=texts,
        task=task or JINA_TASK,
        batch_size=32,
        normalize=True
    )
    return embeddings


def embed_df_with_es(es_client: Elasticsearch,
                     df: pd.DataFrame,
                     model_id: str = None) -> np.ndarray:
    """
    Embed all rows in df using ES inference, adding 'embeddings' column.
    """
    if model_id is None:
        model_id = select_model(es_client)
    tqdm.pandas(desc="Embedding via ES")
    df["text_for_embedding"] = df.apply(combine_text_with_metadata, axis=1)
    df["embeddings"] = df["text_for_embedding"].progress_apply(
        lambda text: get_embedding(text, es_client, model_id)
    )
    arr = np.array(df["embeddings"].tolist())
    print(f"✅ Embeddings shape: {arr.shape}")
    return arr


def load_embedding_model(model_name: str = "intfloat/e5-large-v2") -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model for local embeddings.
    """
    return SentenceTransformer(model_name)


def compute_embeddings(model: SentenceTransformer,
                       texts: list,
                       batch_size: int = 32,
                       normalize: bool = True) -> np.ndarray:
    """
    Compute embeddings for a list of texts locally.
    """
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    arr = np.array(embs)
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        arr = arr / norms
    return arr
