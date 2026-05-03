# Docker Local Runbook

This project can run locally in Docker without copying private data or secrets into the image.

## 1. Prepare `.env`

Copy the template and fill in real values locally:

```bash
cp .env.template .env
```

Required for the Streamlit app:

```bash
ES_URL="https://your-elasticsearch-cluster.com:9200"
ES_API_KEY="your_api_key"
INDEX_NAME="your_kb_index"
EMBEDDING_FIELD="jina_embeddings_v3"
KB_BASE_URL="https://your-kb-base-url/view/"
```

Use `ES_USER` and `ES_PASSWORD` instead of `ES_API_KEY` only if your cluster allows basic auth.

For the optional local Elasticsearch node below, use:

```bash
ES_URL="http://elasticsearch:9200"
INDEX_NAME="kb_articles_metadata_jina_v3"
EMBEDDING_FIELD="jina_embeddings_v3"
```

Because the local development node disables Elasticsearch security, do not set `ES_API_KEY`, `ES_USER`, or `ES_PASSWORD` for that local-only path.

## 2. Optional Local Data Mount

Private CSV exports are intentionally ignored by git and Docker image builds. If you want to run ingestion, place data under `data/` locally and set:

```bash
SOURCE_CSV_PATH="data/input.csv"
```

The compose app mounts `./data` read-only at `/app/data`.

## 3. Build and Run

```bash
docker compose up --build
```

Open:

```bash
http://localhost:8501
```

The first run can be slow because Jina and transformer models may download into the `model-cache` Docker volume.

## 4. Optional Local Elasticsearch

Compose includes an optional single-node Elasticsearch service for local checks:

```bash
docker compose --profile elasticsearch up -d elasticsearch
docker compose --profile elasticsearch up -d duplicate-detection
```

The node is configured with:

- Elasticsearch `8.19.6`
- `ES_JAVA_OPTS=-Xms4g -Xmx4g`
- `mem_limit: 5g`
- persistent Docker volume `es-data`
- `xpack.security.enabled=false` for local development only

Docker Desktop must have enough memory available. Set Docker Desktop's memory limit to at least 6 GB, preferably 8 GB, before starting this profile.

Check health:

```bash
curl http://localhost:9200/_cluster/health?pretty
```

## 5. Run Tests in Docker

```bash
docker compose --profile test run --rm tests
```

## Data Safety

The Docker image excludes `.env`, CSV/TSV/XLSX files, zip archives, notebooks, caches, and `data/`. Runtime secrets are injected through `env_file: .env`; runtime data is mounted from `./data`.
