# Jina AI Duplicate Detection - Quick Overview

## What This Does
Advanced KB article duplicate detection using **Jina AI embeddings** with **16x better context understanding** than previous models.

## Key Improvements
- **Context**: 8,192 tokens (vs 512 tokens previously)
- **Accuracy**: 20-30% better precision with AI reranking
- **Speed**: 2-5 seconds per search with model caching
- **Features**: Interactive web UI, clustering, A/B testing

## Quick Start
1. Copy `.env.template` to `.env` and configure your Elasticsearch details
2. `pip install -r requirements.txt`
3. `streamlit run run_pipeline.py`

## Data Safety
CSV exports, notebooks, archives, and `.env` files are intentionally ignored by git because they can contain private KB content, customer references, or credentials. Set `SOURCE_CSV_PATH` in `.env` when running ingestion locally.

## Architecture
```
Query → Jina AI Embedding → Hybrid Search → AI Reranking → Clustered Results
```

For detailed setup and usage, see [README.md](README.md)
