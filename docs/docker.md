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

## 4. Run Tests in Docker

```bash
docker compose --profile test run --rm tests
```

## Data Safety

The Docker image excludes `.env`, CSV/TSV/XLSX files, zip archives, notebooks, caches, and `data/`. Runtime secrets are injected through `env_file: .env`; runtime data is mounted from `./data`.
