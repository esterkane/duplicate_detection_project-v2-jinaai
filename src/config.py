# src/config.py

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from .env file
load_dotenv()

# --- Elasticsearch Configuration ---
# Use environment variables for security
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")  # Optional: If using user/pass
ES_PASSWORD = os.getenv("ES_PASSWORD")  # Optional: If using user/pass

# Updated for Jina AI migration
INDEX_NAME = os.getenv("INDEX_NAME", "kb_articles_metadata_jina_v3")
ES_MODEL_ID = os.getenv("ES_MODEL_ID", "intfloat__e5-large-v2_search")  # Keep as fallback
EMBEDDING_FIELD = os.getenv("EMBEDDING_FIELD", "jina_embeddings_v3")

# Add Jina AI configuration
JINA_MODEL_NAME = os.getenv("JINA_MODEL_NAME", "jinaai/jina-embeddings-v3")
JINA_USE_API = os.getenv("JINA_USE_API", "False").lower() == "true"
JINA_API_KEY = os.getenv("JINA_API_KEY")  # For API usage
JINA_MAX_LENGTH = int(os.getenv("JINA_MAX_LENGTH", "8192"))
JINA_DIMENSIONS = int(os.getenv("JINA_DIMENSIONS", "1024"))
JINA_TASK = os.getenv("JINA_TASK", "text-matching")  # Optimized for duplicate detection

KB_BASE_URL = os.getenv("KB_BASE_URL", "https://support.elastic.dev/knowledge/view/")


def validate_elasticsearch_config() -> None:
    """Validate connection settings before creating an Elasticsearch client."""
    if not ES_URL and not ES_CLOUD_ID:
        raise ValueError("Either ES_URL or ES_CLOUD_ID environment variable is required")
    if not ES_API_KEY and not (ES_USER and ES_PASSWORD):
        raise ValueError("Either ES_API_KEY or ES_USER/ES_PASSWORD must be provided")
