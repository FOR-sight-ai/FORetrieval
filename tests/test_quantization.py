from __future__ import annotations

import pytest
import torch
from transformers import BitsAndBytesConfig

from foretrieval.colpali import ColPaliModel
from foretrieval.retriever import MultiModalRetrieverModel


class _DummyModel:
    def eval(self):
        return self

    def to(self, _device):
        return self


class _DummyProcessor:
    pass


def test_quant_modes_mutually_exclusive():
    with pytest.raises(ValueError, match="Only one quantization mode"):
        ColPaliModel.from_pretrained(
            "vidore/colpali-v1.3",
            device="cpu",
            load_in_4bit=True,
            load_in_8bit=True,
        )


def test_invalid_4bit_compute_dtype(monkeypatch):
    import foretrieval.colpali as colpali_module

    monkeypatch.setattr(colpali_module.importlib.util, "find_spec", lambda _name: object())

    with pytest.raises(ValueError, match="Invalid bnb_4bit_compute_dtype"):
        ColPaliModel.from_pretrained(
            "vidore/colpali-v1.3",
            device="cpu",
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float64",
        )


def test_wrapper_forwards_quant_args(monkeypatch):
    import foretrieval.retriever as retriever_module

    captured = {}

    def _fake_from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(retriever_module.ColPaliModel, "from_pretrained", _fake_from_pretrained)

    _ = MultiModalRetrieverModel.from_pretrained(
        "vidore/colpali-v1.3",
        device="cpu",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    assert captured["args"][0] == "vidore/colpali-v1.3"
    assert captured["kwargs"]["load_in_4bit"] is True
    assert captured["kwargs"]["load_in_8bit"] is False
    assert captured["kwargs"]["bnb_4bit_quant_type"] == "nf4"
    assert captured["kwargs"]["bnb_4bit_compute_dtype"] == "float16"


@pytest.mark.parametrize(
    ("loader_attr", "quant_kwargs", "expected"),
    [
        (
            "ColPali",
            {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"},
            {"load_in_4bit": True, "load_in_8bit": False, "dtype": torch.float16},
        ),
        (
            "ColPali",
            {"load_in_8bit": True},
            {"load_in_4bit": False, "load_in_8bit": True, "dtype": None},
        ),
    ],
)
def test_quantization_config_is_passed(monkeypatch, loader_attr, quant_kwargs, expected):
    import foretrieval.colpali as colpali_module

    monkeypatch.setattr(colpali_module.importlib.util, "find_spec", lambda _name: object())

    captured = {}

    def _fake_model_from_pretrained(*_args, **kwargs):
        captured["model_kwargs"] = kwargs
        return _DummyModel()

    def _fake_processor_from_pretrained(*_args, **_kwargs):
        return _DummyProcessor()

    monkeypatch.setattr(getattr(colpali_module, loader_attr), "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(colpali_module.ColPaliProcessor, "from_pretrained", _fake_processor_from_pretrained)

    _ = ColPaliModel.from_pretrained("vidore/colpali-v1.3", device="cpu", **quant_kwargs)

    qcfg = captured["model_kwargs"]["quantization_config"]
    assert isinstance(qcfg, BitsAndBytesConfig)
    assert qcfg.load_in_4bit is expected["load_in_4bit"]
    assert qcfg.load_in_8bit is expected["load_in_8bit"]
    if expected["dtype"] is not None:
        assert qcfg.bnb_4bit_compute_dtype == expected["dtype"]


def test_quantization_requires_bitsandbytes(monkeypatch):
    import foretrieval.colpali as colpali_module

    monkeypatch.setattr(colpali_module.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ImportError, match="bitsandbytes"):
        ColPaliModel.from_pretrained(
            "vidore/colpali-v1.3",
            device="cpu",
            load_in_4bit=True,
        )
