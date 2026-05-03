# KB Article Duplicate Detection with Jina AI

## Overview

This is an advanced Knowledge Base article duplicate detection system powered by **Jina AI embeddings** and **AI reranking**. The system provides a 16x improvement in context understanding compared to traditional models, delivering superior semantic search and duplicate detection capabilities.

### Key Features

- 🚀 **Jina AI v3 Embeddings**: 8,192 token context window (16x improvement over e5-large-v2)
- 🎯 **AI-Powered Reranking**: Cross-encoder reranking for 20-30% better precision
- 🔍 **Hybrid Search**: k-NN semantic search + keyword search with Reciprocal Rank Fusion (RRF)
- 📊 **Advanced Clustering**: HDBSCAN clustering with optimized parameters for technical content
- 📈 **A/B Testing**: Compare baseline vs reranked results side-by-side
- 🎨 **Interactive Visualization**: UMAP 2D projections of semantic clusters
- ⚡ **Performance Optimized**: Model caching and GPU acceleration (Apple Silicon MPS)

## Architecture

The system uses a two-stage retrieval approach:
1. **Stage 1**: Fast hybrid search retrieves top candidates from Elasticsearch
2. **Stage 2**: Jina AI cross-encoder reranks candidates for optimal precision

## Project Structure

```
duplicate_detection_project/
├── src/                          # Core application logic
│   ├── config.py                 # Configuration management
│   ├── es_client.py              # Elasticsearch connection
│   ├── embeddings_jina.py        # Jina AI embedding & reranking models
│   ├── search_jina.py            # Enhanced search with reranking
│   ├── hits_analysis.py          # Result processing & clustering
│   ├── deduplication.py          # HDBSCAN clustering algorithms
│   └── ingest.py                 # Data ingestion pipeline
├── examples/                     # Demo scripts and examples
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks for analysis
├── run_pipeline.py               # Main Streamlit application
├── test_jina_search.py          # Jina AI search testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Prerequisites

- **Python 3.11+** (recommended for optimal performance)
- **Elasticsearch 8.x** with:
  - Accessible via URL and API Key
  - Index containing KB articles with dense vector field
- **Hardware**: 
  - 8GB+ RAM recommended
  - GPU optional but recommended (Apple Silicon MPS supported)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd duplicate_detection_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Elasticsearch Configuration
ES_URL="https://your-elasticsearch-cluster.com:9200"
ES_API_KEY="your_base64_api_key"

# Optional: Alternative authentication
# ES_USER="your_username"
# ES_PASSWORD="your_password"

# Index Configuration
INDEX_NAME="your_kb_index_name"
EMBEDDING_FIELD="your_embedding_field_name"
KB_BASE_URL="https://your-kb-base-url.com/view/"
```

### 3. Run the Application

```bash
streamlit run run_pipeline.py
```

Navigate to `http://localhost:8501` to access the web interface.

### 4. Run with Docker

Docker is the easiest way to check the app locally with all native ML dependencies installed:

```bash
cp .env.template .env
# Edit .env with your Elasticsearch settings
docker compose up --build
```

Then open `http://localhost:8501`.

See [docs/docker.md](docs/docker.md) for Docker details, data mounting, test commands, and data-safety notes.

## Usage Guide

### Basic Search
1. Enter your search query (e.g., "Elasticsearch cluster configuration")
2. Adjust search parameters in the sidebar:
   - **Number of Results**: How many final results to return
   - **Reranking Candidates**: How many candidates to rerank (more = better quality)
   - **Text/k-NN Boosts**: Balance between keyword and semantic search
3. Enable **AI Reranking** for optimal results
4. Click "Search Similar Articles"

### Advanced Features

#### A/B Testing
- Enable "Show A/B Comparison" to compare baseline vs reranked results
- Observe how reranking improves result relevance and ordering

#### Clustering Analysis
- View semantic clusters of search results
- Identify potential duplicates and related content groups
- Explore 2D UMAP visualization of article relationships

HDBSCAN clustering is documented in [docs/hdbscan.md](docs/hdbscan.md), including installation requirements, fallback behavior, label meanings, and tuning guidance.

#### Performance Tuning
- Adjust reranking candidate count for speed vs quality tradeoff
- Modify search boosts for different content types
- Use clustering parameters for different granularity levels

## Configuration

Key settings in `src/config.py`:

```python
# Model Configuration
JINA_MODEL_NAME = "jinaai/jina-embeddings-v3"
JINA_TASK = "text-matching"  # Optimized for duplicate detection

# Search Configuration
INDEX_NAME = "kb_articles_metadata_jina_v3"
EMBEDDING_FIELD = "jina_embeddings_v3"

# Application Settings
KB_BASE_URL = "https://support.elastic.dev/knowledge/view/"
```

## Data Ingestion

To ingest new data with Jina AI embeddings:

```bash
# Prepare your CSV data file outside git and set SOURCE_CSV_PATH in .env
# Run ingestion pipeline
python src/ingest.py
```

The ingestion pipeline:
1. Processes CSV data in chunks
2. Generates Jina AI embeddings (8,192 token context)
3. Creates optimized Elasticsearch index
4. Handles memory management for large datasets

## Performance

### Benchmark Results
- **Context Window**: 8,192 tokens (16x improvement over e5-large-v2)
- **Reranking Precision**: 20-30% improvement over baseline
- **Search Speed**: ~2-5 seconds for reranked results (with cached models)
- **Memory Usage**: Optimized for Apple Silicon MPS acceleration

### Optimization Tips
- Enable model caching in production (handled automatically)
- Use appropriate reranking candidate counts (50-100 recommended)
- Leverage GPU acceleration when available
- Monitor Elasticsearch cluster performance

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/

# Dockerized unit tests
docker compose --profile test run --rm tests

# Integration test for Jina AI search
python test_jina_search.py

# Example usage
python examples/jina_comparison_demo.py
```

## Development

### Adding New Features
1. Follow the modular architecture in `src/`
2. Use type hints and comprehensive docstrings
3. Add corresponding tests in `tests/`
4. Update documentation as needed

### Extending Search Capabilities
- Modify `src/search_jina.py` for new search algorithms
- Add new reranking models in `src/embeddings_jina.py`
- Customize clustering in `src/deduplication.py`

## Troubleshooting

### Common Issues

**Memory Errors (Apple Silicon)**
```bash
# Reduce batch sizes in config.py
EMBEDDING_BATCH_SIZE = 4  # Reduce from 8
RERANK_BATCH_SIZE = 8     # Reduce from 16
```

**Slow Performance**
- Reduce reranking candidates
- Check Elasticsearch cluster health
- Verify model caching is working

**Connection Issues**
- Verify `.env` configuration
- Test Elasticsearch connectivity
- Check API key permissions

## License

This project is part of the Elastic Support AI Tools collection.

## Contributing

Please follow Elastic's contribution guidelines and ensure all tests pass before submitting pull requests.

---

**🚀 Powered by Jina AI v3** | **📊 16x Context Improvement** | **🎯 AI-Enhanced Precision**
