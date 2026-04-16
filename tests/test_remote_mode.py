from __future__ import annotations

import torch

from foretrieval.colpali import ColPaliModel
from foretrieval.retriever import MultiModalRetrieverModel


def test_wrapper_forwards_remote_mode_args(monkeypatch):
    import foretrieval.retriever as retriever_module

    captured = {}

    def _fake_from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        retriever_module.ColPaliModel, "from_pretrained", _fake_from_pretrained
    )

    _ = MultiModalRetrieverModel.from_pretrained(
        "vidore/colqwen2-v1.0",
        embedding_mode="remote",
        embedding_server_url="http://localhost:8000",
        embedding_server_token="abc",
        embedding_request_timeout=12.0,
    )

    assert captured["kwargs"]["embedding_mode"] == "remote"
    assert captured["kwargs"]["embedding_server_url"] == "http://localhost:8000"
    assert captured["kwargs"]["embedding_server_token"] == "abc"
    assert captured["kwargs"]["embedding_request_timeout"] == 12.0


def test_remote_mode_requires_server_url(monkeypatch):
    import foretrieval.colpali as colpali_module

    with monkeypatch.context() as m:
        m.setattr(colpali_module.importlib.util, "find_spec", lambda _name: None)
        try:
            ColPaliModel.from_pretrained("vidore/colqwen2-v1.0", embedding_mode="remote")
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "embedding_server_url" in str(exc)


def test_embed_queries_uses_remote_client(monkeypatch):
    import foretrieval.colpali as colpali_module

    class _DummyProcessor:
        pass

    class _FakeRemoteClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode_queries(self, queries):
            assert queries == ["hello"]
            return torch.randn(1, 4, 8)

        def encode_images(self, _images):
            return torch.randn(1, 4, 8)

    monkeypatch.setattr(colpali_module, "RemoteEmbeddingClient", _FakeRemoteClient)
    monkeypatch.setattr(
        colpali_module.ColQwen2Processor,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyProcessor(),
    )

    model = ColPaliModel.from_pretrained(
        "vidore/colqwen2-v1.0",
        embedding_mode="remote",
        embedding_server_url="http://localhost:8000",
        device="cpu",
    )
    out = model.encode_query("hello")
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1
