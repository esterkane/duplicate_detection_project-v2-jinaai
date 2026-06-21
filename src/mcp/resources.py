"""Lazily-constructed resource singletons for the MCP layer.

The MCP server is a sibling front-end to the Streamlit app: both adapt the same
core (Elasticsearch hybrid retrieval, the Jina embedder, the similarity helper)
to a transport. These accessors reuse :mod:`src.config`, :func:`src.es_client.
get_es_client`, and :func:`src.embeddings_jina.load_jina_embedding_model`, so the
MCP layer never reimplements connection or model-loading logic.

The Elasticsearch client and the Jina embedder are expensive to build (the
embedder loads a local model — the README warns the first query can take
30-60s), so each is cached for the process lifetime and built on first use. A
tool that never needs an embedder (``get_chunk``) never pays to load one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .. import config
from ..embeddings_jina import load_jina_embedding_model
from ..es_client import get_es_client as _get_es_client


@lru_cache(maxsize=1)
def get_es_client() -> Any:
    """Cached Elasticsearch client built from the configured credentials."""
    return _get_es_client()


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Cached Jina embedder (local model or API), per :mod:`src.config`."""
    return load_jina_embedding_model(
        model_name=config.JINA_MODEL_NAME,
        use_api=config.JINA_USE_API,
        api_key=config.JINA_API_KEY,
    )


def get_index_name() -> str:
    """The Elasticsearch index the tools search against."""
    return config.INDEX_NAME


def get_embedding_field() -> str:
    """The dense-vector field name used for kNN search."""
    return config.EMBEDDING_FIELD


def get_kb_base_url() -> str:
    """Base URL used to construct a human-openable article link."""
    return config.KB_BASE_URL
