# Agent Access — read-only MCP server

The project exposes its retrieval / duplicate-detection core as a small set of
**read-only** [Model Context Protocol](https://modelcontextprotocol.io) tools so
that an MCP client (Claude Code, Cursor, a custom agent) can search the knowledge
base, find near-duplicates, and read individual articles.

The server is a **thin adapter**: every tool validates its inputs and forwards to
an existing function in `src/` (`hybrid_search_with_reranking`,
`detect_duplicates`, `get_article_by_id`). No business logic lives in the MCP
layer, and **no tool mutates anything** — ingestion is intentionally *not*
exposed.

## Package layout

```
src/mcp/
├── errors.py       # typed ToolError + guard -> structured error payloads
├── tools.py        # pure, importable *_impl functions (unit-tested with fakes)
├── resources.py    # lazy, cached ES client / Jina embedder singletons (reuse src/config.py)
└── server.py       # FastMCP "duplicate-detection" — registers the 3 tools
```

`tools.py` has no FastMCP/HTTP coupling: each `*_impl` takes its dependencies
(`es_client`, `embedder`, `index_name`, …) as explicit arguments, so it is unit
tested directly against a fake Elasticsearch and a fake embedder
(`tests/test_mcp_tools.py`). `server.py` supplies the real cached singletons.

## Tools

All three tools are read-only.

### `hybrid_search(query, filters?, k=10)`

Hybrid kNN + keyword (RRF-fused) search over the KB index. Wraps
`src.search_jina.hybrid_search_with_reranking`.

- `query` (str, required): natural-language query.
- `filters` (object, optional): case-insensitive substring filters on returned
  fields, e.g. `{"title": "kibana"}` or `{"products": "Elasticsearch"}`.
- `k` (int, 1..100, default 10): number of results.

Returns `{query, reranked, count, results: [{chunk_id, article_id, title,
summary, body_preview, products, score, url}]}`.

### `find_duplicates(text?, chunk_id?, threshold=0.9)`

Near-duplicate articles for a seed. Provide **exactly one** of `text` or
`chunk_id`. Gathers candidates via hybrid search, then applies the project's
dependency-free `src.deduplication.detect_duplicates` pairwise string-similarity
helper at `threshold` (0.0..1.0; `0.9` is the project's tuned default).

Returns `{seed, threshold, count, duplicates: [{score, candidate}]}` ordered by
descending similarity.

### `get_chunk(chunk_id)`

Fetch a single article (full body) by its `article_id`. Wraps
`src.search.get_article_by_id`.

Returns `{chunk: {chunk_id, article_id, title, summary, body, products, url}}`.

## Error contract

Tools never raise or leak a stack trace. Expected failures return a structured
payload:

```json
{
  "isError": true,
  "errorCategory": "validation | transient | permission | business",
  "isRetryable": true,
  "message": "human-readable summary",
  "details": {}
}
```

| Category     | Meaning                                              | Retryable |
|--------------|------------------------------------------------------|-----------|
| `validation` | Bad input (empty query, out-of-range threshold/k)    | no        |
| `business`   | Valid request that can't be satisfied (unknown id)   | no        |
| `permission` | Caller not permitted                                 | no        |
| `transient`  | Backend (Elasticsearch / Jina) momentarily down      | yes       |

Note: `hybrid_search` inherits the repo's graceful fallback — if the underlying
`hybrid_search_with_reranking` cannot reach Elasticsearch it returns an **empty**
result set (`count: 0`) rather than a transient error. The non-swallowing
`get_chunk` / `find_duplicates`-by-`chunk_id` fetch path *does* surface a
retryable `transient` error.

## Running the server

From the repository root, with dependencies installed (`pip install -r
requirements.txt`) and a valid `.env` (see `.env.template` — `ES_URL`,
`ES_API_KEY`, `INDEX_NAME`, …):

```bash
python -m src.mcp.server
```

Transport is selected by `MCP_TRANSPORT` (default `stdio`; set `http` for
streamable-HTTP). The first call constructs the ES client and loads the Jina
embedder lazily — the README's "first query can take 30-60s" warning applies to
the first tool call too.

> The server needs a live Elasticsearch index and the Jina model (local or API).
> The unit tests do **not** — they run entirely against fakes.

## Client registration

Example Claude Code / Claude Desktop entry (`mcp.json` / `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "duplicate-detection": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/duplicate_detection_project-v2-jinaai",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

`.env` in the project root supplies the Elasticsearch / Jina credentials.

## Example calls and outputs

`hybrid_search("kibana alerting", k=2)`:

```json
{
  "query": "kibana alerting",
  "reranked": false,
  "count": 2,
  "results": [
    {
      "chunk_id": "kb-001",
      "article_id": "kb-001",
      "title": "Configure Kibana alerting",
      "summary": "How to set up rules and connectors...",
      "body_preview": "Alerting in Kibana lets you...",
      "products": ["Kibana"],
      "score": 18.42,
      "url": "https://support.elastic.dev/knowledge/view/kb-001"
    }
  ]
}
```

`find_duplicates(chunk_id="kb-001", threshold=0.85)`:

```json
{
  "seed": { "chunk_id": "kb-001", "title": "Configure Kibana alerting", "...": "..." },
  "threshold": 0.85,
  "count": 1,
  "duplicates": [
    { "score": 0.91, "candidate": { "chunk_id": "kb-077", "title": "Set up Kibana alerts", "...": "..." } }
  ]
}
```

`get_chunk("does-not-exist")` (business error):

```json
{
  "isError": true,
  "errorCategory": "business",
  "isRetryable": false,
  "message": "No article found with chunk_id 'does-not-exist'.",
  "details": { "chunk_id": "does-not-exist" }
}
```

## Tests

```bash
pytest tests/test_mcp_tools.py
```

The MCP unit tests use fake Elasticsearch / embedder objects — no live
Elasticsearch, no Jina API, no model download. Live Elasticsearch + Jina is an
**integration** concern and must be exercised locally.
