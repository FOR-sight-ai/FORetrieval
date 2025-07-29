from typing import Generator

import pytest
from colpali_engine.models import ColPali
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

from foretrieval import MultiModalRetrieverModel
from foretrieval.colpali import ColPaliModel


@pytest.fixture(scope="module")
def colpali_rag_model() -> Generator[MultiModalRetrieverModel, None, None]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")
    yield MultiModalRetrieverModel.from_pretrained("vidore/colpali-v1.2", device=device)
    tear_down_torch()


@pytest.mark.slow
def test_load_colpali_from_pretrained(colpali_rag_model: MultiModalRetrieverModel):
    assert isinstance(colpali_rag_model, MultiModalRetrieverModel)
    assert isinstance(colpali_rag_model.model, ColPaliModel)
    assert isinstance(colpali_rag_model.model.model, ColPali)
