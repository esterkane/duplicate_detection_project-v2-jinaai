from src.search import knn_search


class FakeElasticsearch:
    def __init__(self):
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"hits": {"hits": [{"_id": "1"}]}}


def test_knn_search_uses_requested_embedding_field():
    client = FakeElasticsearch()

    hits = knn_search(
        es_client=client,
        query_vector=[0.1, 0.2],
        user_query="cluster settings",
        index_name="kb",
        embedding_field="custom_vector",
    )

    body = client.search_calls[0]["body"]
    assert hits == [{"_id": "1"}]
    assert body["knn"]["field"] == "custom_vector"
    assert "custom_vector" in body["_source"]


def test_knn_search_returns_empty_list_on_client_error():
    class FailingElasticsearch:
        def search(self, **kwargs):
            raise RuntimeError("boom")

    assert knn_search(
        es_client=FailingElasticsearch(),
        query_vector=[0.1],
        user_query="query",
        index_name="kb",
        embedding_field="vector",
    ) == []
