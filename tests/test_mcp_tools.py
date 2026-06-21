"""Unit tests for the read-only MCP tool implementations.

These exercise ``src.mcp.tools`` against FAKE Elasticsearch and a FAKE embedder —
no live Elasticsearch, no Jina API, no model downloads — mirroring the existing
``FakeElasticsearch`` pattern in ``tests/test_search.py``. Coverage per tool:
success shape, validation error, unknown-id / no-results business case, and a
transient backend error.
"""

import unittest

from src.mcp.tools import (
    find_duplicates_impl,
    get_chunk_impl,
    hybrid_search_impl,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Returns a fixed-length vector regardless of input."""

    def encode(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _hit(article_id, title, summary="", body=""):
    return {
        "_id": article_id,
        "_score": 1.23,
        "_source": {
            "article_id": article_id,
            "content_title": title,
            "content_summary": summary,
            "content_body": body,
            "metadata_products": ["Elasticsearch"],
        },
    }


class FakeElasticsearch:
    """Fake ES that returns canned hits for kNN searches and term look-ups."""

    def __init__(self, search_hits=None, by_id=None):
        self._search_hits = search_hits if search_hits is not None else []
        self._by_id = by_id or {}
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        body = kwargs.get("body", {})
        # A term query on article_id is a get-by-id look-up.
        term = body.get("query", {}).get("term")
        if term and "article_id" in term:
            article_id = term["article_id"]
            hit = self._by_id.get(article_id)
            return {"hits": {"hits": [hit] if hit else []}}
        return {"hits": {"hits": self._search_hits}}


class FailingElasticsearch:
    """Raises an Elasticsearch ConnectionError-shaped error on any search."""

    def search(self, **kwargs):
        from elasticsearch import ConnectionError as ESConnectionError

        raise ESConnectionError("backend down")


# --------------------------------------------------------------------------- #
# hybrid_search
# --------------------------------------------------------------------------- #
class TestHybridSearch(unittest.TestCase):
    def _run(self, es, query="kibana alerts", **kw):
        return hybrid_search_impl(
            query,
            es_client=es,
            embedder=FakeEmbedder(),
            index_name="kb",
            embedding_field="vec",
            kb_base_url="https://kb.example/view/",
            **kw,
        )

    def test_success_shape(self):
        es = FakeElasticsearch(search_hits=[_hit("a1", "Kibana alerts"), _hit("a2", "Other")])
        result = self._run(es)

        self.assertNotIn("isError", result)
        self.assertEqual(result["count"], 2)
        first = result["results"][0]
        self.assertEqual(first["chunk_id"], "a1")
        self.assertEqual(first["url"], "https://kb.example/view/a1")
        self.assertIn("score", first)
        # Raw ES internals are not leaked.
        self.assertNotIn("_source", first)

    def test_validation_error_on_empty_query(self):
        es = FakeElasticsearch(search_hits=[])
        result = self._run(es, query="   ")

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "validation")
        self.assertFalse(result["isRetryable"])

    def test_filters_narrow_results(self):
        es = FakeElasticsearch(search_hits=[_hit("a1", "Kibana alerts"), _hit("a2", "Logstash pipeline")])
        result = self._run(es, filters={"title": "kibana"})

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["chunk_id"], "a1")

    def test_backend_error_returns_empty(self):
        # The existing hybrid_search_with_reranking swallows ES errors and returns
        # [] (a deliberate graceful-fallback invariant in this repo), so a backend
        # failure here surfaces as an empty, non-error result rather than a
        # transient error. The get_chunk path, which uses a non-swallowing fetch,
        # does surface a transient error (see TestGetChunk).
        result = self._run(FailingElasticsearch())

        self.assertNotIn("isError", result)
        self.assertEqual(result["count"], 0)


# --------------------------------------------------------------------------- #
# find_duplicates
# --------------------------------------------------------------------------- #
class TestFindDuplicates(unittest.TestCase):
    def _run(self, es, **kw):
        return find_duplicates_impl(
            es_client=es,
            embedder=FakeEmbedder(),
            index_name="kb",
            embedding_field="vec",
            kb_base_url="https://kb.example/view/",
            **kw,
        )

    def test_success_finds_duplicate_by_text(self):
        es = FakeElasticsearch(
            search_hits=[
                _hit("dup", "Elastic Search cluster setup"),
                _hit("other", "Completely unrelated topic about cats"),
            ]
        )
        result = self._run(es, text="elastic search cluster setup", threshold=0.8)

        self.assertNotIn("isError", result)
        self.assertGreaterEqual(result["count"], 1)
        self.assertEqual(result["duplicates"][0]["candidate"]["chunk_id"], "dup")

    def test_validation_error_when_no_seed(self):
        es = FakeElasticsearch(search_hits=[])
        result = self._run(es)

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "validation")

    def test_validation_error_when_both_seeds(self):
        es = FakeElasticsearch(search_hits=[])
        result = self._run(es, text="x", chunk_id="a1")

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "validation")

    def test_business_error_unknown_chunk_id(self):
        es = FakeElasticsearch(search_hits=[], by_id={})
        result = self._run(es, chunk_id="missing")

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "business")
        self.assertFalse(result["isRetryable"])

    def test_no_candidates_returns_empty(self):
        es = FakeElasticsearch(search_hits=[])
        result = self._run(es, text="anything")

        self.assertNotIn("isError", result)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["duplicates"], [])


# --------------------------------------------------------------------------- #
# get_chunk
# --------------------------------------------------------------------------- #
class TestGetChunk(unittest.TestCase):
    def _run(self, es, chunk_id="a1"):
        return get_chunk_impl(
            chunk_id,
            es_client=es,
            index_name="kb",
            kb_base_url="https://kb.example/view/",
        )

    def test_success_shape(self):
        es = FakeElasticsearch(by_id={"a1": _hit("a1", "Title", "Summary", "Full body text")})
        result = self._run(es)

        self.assertNotIn("isError", result)
        chunk = result["chunk"]
        self.assertEqual(chunk["chunk_id"], "a1")
        self.assertEqual(chunk["body"], "Full body text")
        self.assertNotIn("body_preview", chunk)

    def test_validation_error_empty_id(self):
        es = FakeElasticsearch()
        result = self._run(es, chunk_id="")

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "validation")

    def test_business_error_unknown_id(self):
        es = FakeElasticsearch(by_id={})
        result = self._run(es, chunk_id="nope")

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "business")

    def test_transient_backend_error(self):
        result = self._run(FailingElasticsearch())

        self.assertTrue(result["isError"])
        self.assertEqual(result["errorCategory"], "transient")
        self.assertTrue(result["isRetryable"])


if __name__ == "__main__":
    unittest.main()
