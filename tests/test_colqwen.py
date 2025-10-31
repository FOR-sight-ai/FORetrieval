from typing import Generator

import pytest
from colpali_engine.models import ColQwen2_5
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

from foretrieval import MultiModalRetrieverModel
from foretrieval.colpali import ColPaliModel


@pytest.fixture(scope="module")
def colqwen_rag_model() -> Generator[MultiModalRetrieverModel, None, None]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")
    yield MultiModalRetrieverModel.from_pretrained(
        "vidore/colqwen2.5-v0.2", device=device
    )
    tear_down_torch()


@pytest.mark.slow
def test_load_colqwen_from_pretrained(colqwen_rag_model: MultiModalRetrieverModel):
    assert isinstance(colqwen_rag_model, MultiModalRetrieverModel)
    assert isinstance(colqwen_rag_model.model, ColPaliModel)
    assert isinstance(colqwen_rag_model.model.model, ColQwen2_5)
