# CLAUDE.md — duplicate_detection_project-v2-jinaai

Knowledge-base article duplicate/near-duplicate detection using Jina AI embeddings,
Elasticsearch hybrid retrieval, cross-encoder reranking, and HDBSCAN clustering,
fronted by a Streamlit UI.

## Run / test commands

All commands run from the repository root. There is no `python` guarantee on every
host; use `python3` (or `py` on Windows) / the project venv as appropriate.

```bash
# Install deps
pip install -r requirements.txt

# Run the app (Streamlit UI on http://localhost:8501)
streamlit run run_pipeline.py

# Ingest data (reads SOURCE_CSV_PATH from .env, writes a Jina-embedded ES index)
python src/ingest.py

# Unit tests (pytest config in pytest.ini; collects tests/test_*.py)
pytest tests/

# Integration smoke test (requires a live Elasticsearch + Jina model)
python test_jina_search.py

# Demo
python examples/jina_comparison_demo.py
```

### Docker

```bash
cp .env.template .env          # then edit ES_URL / ES_API_KEY / INDEX_NAME
docker compose up --build      # app only -> http://localhost:8501

# Optional local single-node Elasticsearch (4g heap), opt-in via profile
docker compose --profile elasticsearch up --build

# Dockerized unit tests
docker compose --profile test run --rm tests
```

See `docs/docker.md` and `docs/hdbscan.md` for details.

### Lint / type-check / CI

- **No linter, formatter, or type-checker is configured** (no ruff/flake8/black,
  no mypy, no `pyproject.toml`/`setup.cfg`/`Makefile`). Do not invent one.
- **No CI** (no `.github/workflows`). The only automated quality gate is `pytest`.

## Architecture in 5 lines

1. `src/ingest.py` reads a CSV, generates Jina v3 embeddings (`src/embeddings_jina.py`),
   and writes a dense-vector Elasticsearch index (default `kb_articles_metadata_jina_v3`).
2. A query is embedded, then `src/search_jina.py` runs a two-stage retrieval:
   hybrid k-NN + keyword search fused with Elasticsearch RRF (stage 1).
3. Stage 2 reranks the top candidates with a Jina cross-encoder for precision
   (toggleable for A/B baseline-vs-reranked comparison).
4. `src/deduplication.py` clusters result embeddings with HDBSCAN (DBSCAN fallback)
   plus a dependency-free string-similarity `detect_duplicates` helper.
5. `run_pipeline.py` is the Streamlit front end; `src/visualization.py` renders UMAP
   2D projections of the clusters; `src/config.py` loads all settings from env/`.env`.

## Invariants I must never break

1. **Deterministic embed -> retrieve -> rerank -> cluster path.** The pipeline is
   embed query -> hybrid search (RRF) -> optional cross-encoder rerank -> HDBSCAN.
   `detect_duplicates` and the HDBSCAN wrapper are deterministic for a given input;
   keep them so. Preserve the existing graceful fallbacks (HDBSCAN -> DBSCAN ->
   all-noise when libs are missing; rerank failure -> baseline hits) rather than
   letting these paths raise.
2. **Quality gate stays green.** `pytest tests/` must pass. Every new module in
   `src/` should get a matching `tests/test_*.py`. Tests must not require network,
   a live Elasticsearch, or real model downloads — existing tests stub the ES client
   (`FakeElasticsearch`) and exercise the dependency-free helpers; keep that pattern.
3. **Hybrid retrieval is required.** Search endpoints combine k-NN (vector) AND
   keyword (`multi_match`) fused via RRF. Do not regress to vector-only or
   keyword-only retrieval.
4. **Provenance / traceability on every result.** Each hit must carry enough to
   explain why it matched: its `_source` (e.g. `article_id`, `content_title`) plus
   its scores. Reranking preserves both `_original_score` and `_rerank_score` —
   do not drop these; matched/duplicate decisions must remain explainable.
5. **No secrets in git.** Credentials (`ES_API_KEY`, `ES_USER`/`ES_PASSWORD`,
   `JINA_API_KEY`) and private KB data load only from `.env` (git/Docker-ignored).
   Use `.env.template` for documentation; never hardcode keys in source,
   `docker-compose.yml`, or committed files. CSV exports / notebooks / archives
   stay out of git.
6. **Repo-specific:** the similarity/cluster thresholds are tuned defaults
   (`detect_duplicates` threshold `0.9`; HDBSCAN `min_cluster_size=2`,
   `cluster_selection_epsilon` ~`0.02-0.05`, `metric='cosine'`). Changing them
   changes which articles count as duplicates — only adjust deliberately, with a
   reason, and validate against `detect_duplicates`'s `0 <= threshold <= 1` guard.
   Keep embeddings reproducible: same model/task (`jina-embeddings-v3`,
   `text-matching`) and dimensions (`1024`) for ingest and query.

## Definition of done

- [ ] `pytest tests/` passes (and new `src/` modules have matching tests).
- [ ] Type checks: **N/A** (no type-checker configured).
- [ ] Quality gate / CI: **N/A** (no CI); pytest is the gate and must be green.
- [ ] Results reproducible: same embedding model/task/dimensions and unchanged
      thresholds give the same retrieval/cluster output; fallbacks still hold.
- [ ] README / `docs/` updated when behavior, commands, or config change.
- [ ] No secrets or private KB data added; new config documented in `.env.template`.
