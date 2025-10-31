from pathlib import Path
from typing import Generator
import shutil
import pytest
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

from foretrieval import MultiModalRetrieverModel

path_document_1 = Path("sample_data/sample_pdf.pdf")
path_document_2 = Path("sample_data/sample_multi_pdf.pdf")
index_root = Path(".test_index")


@pytest.fixture(scope="function")
def rag_model_from_pretrained() -> Generator[MultiModalRetrieverModel, None, None]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")
    yield MultiModalRetrieverModel.from_pretrained(
        "vidore/colqwen2.5-v0.2", device=device
    )
    tear_down_torch()


@pytest.fixture(scope="function")
def rag_model_from_index() -> Generator[MultiModalRetrieverModel, None, None]:
    if index_root.exists():
        # delete the folder
        shutil.rmtree(index_root)
    yield MultiModalRetrieverModel.from_index(
        "multi_doc_index", index_root=index_root.as_posix()
    )
    tear_down_torch()


@pytest.mark.slow
def test_single_pdf(rag_model_from_pretrained: MultiModalRetrieverModel):
    # Index a single PDF
    rag_model_from_pretrained.index(
        input_path=path_document_2,
        index_name="sample_multipage_index",
        store_collection_with_index=True,
        overwrite=True,
    )

    # Test retrieval
    queries = [
        "What is the answer we are looking for on the first page?",
        "What is the answer we are looking for on the second page?",
        "What is the answer we are looking for on the third page?",
    ]

    expected_page = 1
    for query in queries:
        result = rag_model_from_pretrained.search(query, k=1)[0]

        print(f"\nQuery: {query}")
        print(
            f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}"
        )

        # Check if the expected page is in the top results

        assert result.page_num == expected_page, (
            f"Expected page {expected_page} for this query. Got {result.page_num} instead"
        )

        expected_page += 1


@pytest.mark.slow
def test_multi_document(rag_model_from_pretrained: MultiModalRetrieverModel):
    # Index a directory of documents
    rag_model_from_pretrained.index(
        input_path="sample_data/",
        index_name="multi_doc_index",
        store_collection_with_index=True,
        overwrite=True,
    )

    # Test retrieval
    queries = [
        "What is the numerical answer we are looking for in the sample xls file?",
        "What is the numerical answer we are looking for on the third page of the sample multipage pdf file?",
    ]

    for query in queries:
        result = rag_model_from_pretrained.search(query, k=1)[0]

        print(f"\nQuery: {query}")
        print(
            f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}"
        )

        # Check if the expected page (3) is in the top results
        if "third" in query.lower():
            assert result.page_num == 3, "Expected page 3 for multi-page pdf query"
