import logging

from .config import (
    ES_API_KEY,
    ES_CLOUD_ID,
    ES_PASSWORD,
    ES_URL,
    ES_USER,
    validate_elasticsearch_config,
)

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

logger = logging.getLogger(__name__)


def get_es_client():
    """
    Create and return an Elasticsearch client using configured authentication.
    """
    if Elasticsearch is None:
        raise ImportError("The 'elasticsearch' package is required to create an Elasticsearch client")

    validate_elasticsearch_config()

    try:
        logger.info("Connecting to Elasticsearch")
        client_kwargs = {
            "verify_certs": True,
            "request_timeout": 30,
            "retry_on_timeout": True,
            "max_retries": 3,
            "headers": {"Accept": "application/vnd.elasticsearch+json;compatible-with=8"},
        }

        if ES_API_KEY:
            client_kwargs["api_key"] = ES_API_KEY
        else:
            client_kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)

        if ES_CLOUD_ID:
            es_client = Elasticsearch(cloud_id=ES_CLOUD_ID, **client_kwargs)
        else:
            es_client = Elasticsearch(ES_URL, **client_kwargs)

        info = es_client.info()
        logger.info("Successfully connected to Elasticsearch")
        logger.info("Cluster: %s", info.get("cluster_name", "unknown"))
        logger.info("Version: %s", info.get("version", {}).get("number", "unknown"))
        return es_client

    except Exception as e:
        logger.error("Elasticsearch connection failed: %s", e)
        raise ConnectionError(f"Could not establish Elasticsearch connection: {e}")
