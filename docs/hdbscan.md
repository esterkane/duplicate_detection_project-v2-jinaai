# HDBSCAN Notes

HDBSCAN is used after Elasticsearch search to group returned article embeddings into likely duplicate or related clusters.

## What It Does Here

`src.hits_analysis.analyze_knn_hits` extracts valid embedding vectors from search results and calls `cluster_embeddings_hdbscan` in `src.deduplication`.

Labels mean:

- `0`, `1`, `2`, ...: detected clusters.
- `-1`: noise or outlier points.
- `-2`: not clustered because the hit had no usable embedding.

For small result sets, the project uses permissive parameters:

- `min_cluster_size`: usually `2` for small result sets.
- `min_samples`: `1`, so nearby pairs can form clusters.
- `metric`: `cosine`, appropriate for normalized semantic embeddings.
- `cluster_selection_epsilon`: small tolerance for grouping close points.

## Installation

`hdbscan` is listed in `requirements.txt`. On many platforms pip installs a wheel. If pip needs to compile it, native build tooling is required.

Local install:

```bash
pip install -r requirements.txt
```

Docker install:

The `Dockerfile` installs `build-essential`, `g++`, and `libgomp1` before `pip install -r requirements.txt`, which covers the common native extension requirements on Debian slim images.

## Fallback Behavior

The code intentionally imports HDBSCAN lazily and safely:

- If `hdbscan` is installed, real HDBSCAN clustering is used.
- If `hdbscan` is missing but `scikit-learn` is available, it falls back to DBSCAN.
- If both are missing, all embeddings are marked as noise so the app can still show search results.

This fallback is useful for lightweight unit tests, but production/local app checks should install the full requirements so HDBSCAN is active.

## Tuning Guidance

Use smaller values when you want to surface possible duplicate pairs:

```python
min_cluster_size=2
min_samples=1
cluster_selection_epsilon=0.05
```

Use larger values when you want fewer, more conservative clusters:

```python
min_cluster_size=5
min_samples=2
cluster_selection_epsilon=0.0
```

If everything is `-1`, the result set may be too diverse, embeddings may be missing, or the parameters may be too strict. If unrelated articles cluster together, increase `min_cluster_size`, increase `min_samples`, or reduce `cluster_selection_epsilon`.
