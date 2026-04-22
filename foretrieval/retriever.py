from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from PIL import Image

from .colpali import ColPaliModel
from .embedding_server import EmbeddingServerConfig
from .objects import Result
from .models_metadata import MetadataFilter

# Optional langchain integration
try:
    from .integrations import FORetrievalLangChain
except ImportError:
    pass


class MultiModalRetrieverModel:
    """
    Wrapper class for a pretrained multi-modal model, and all the associated utilities.
    Allows you to load a pretrained model from disk or from the hub, build or query an index.

    ## Usage

    Load a pre-trained checkpoint:

    ```python
    from foretrieval import MultiModalRetriever

    RAG = MultiModalRetriever.from_pretrained("vidore/colpali-v1.2")
    ```

    Both methods will load a fully initialised instance of ColPali, which you can use to build and query indexes.

    ```python
    RAG.search("How many people live in France?")
    ```
    """

    model: ColPaliModel

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: str = ".rag_index",
        ingestion: Dict[str, Any] = {"backend": "default"},
        device: str = "cuda",
        verbose: int = 1,
        embedding_server: Optional[EmbeddingServerConfig] = None,
        storage_qdrant: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "float16",
    ):
        """Load a ColPali model from a pre-trained checkpoint.

        Parameters:
            pretrained_model_name_or_path (str): Local path or huggingface model name.
            index_root (str): The root directory where indexes will be stored. Default is ".rag_index".
            ingestion (Dict[str, Any]): Ingestion configuration for the model. Default is {"backend": "default"}.
            device (str): The device to load the model on. Default is "cuda".
            verbose (int): Verbosity level. Default is 1.
            embedding_server (Optional[EmbeddingServerConfig]): If set, embeddings are computed
                on the remote vLLM server instead of locally. Model weights are not loaded locally.
            load_in_4bit (bool): Load model in 4-bit quantization via BitsAndBytes. Requires
                foretrieval[quantization] and a CUDA device. Default False.
            load_in_8bit (bool): Load model in 8-bit quantization via BitsAndBytes. Requires
                foretrieval[quantization] and a CUDA device. Default False.
            bnb_4bit_quant_type (str): 4-bit quantization type, "nf4" or "fp4". Default "nf4".
            bnb_4bit_compute_dtype (str): Compute dtype for 4-bit quant, e.g. "float16". Default "float16".

        Returns:
            cls (MultiModalRetrieverModel): Initialised instance.
        """
        instance = cls()
        instance.model = ColPaliModel.from_pretrained(
            pretrained_model_name_or_path,
            index_root=index_root,
            ingestion=ingestion,
            device=device,
            verbose=verbose,
            embedding_server=embedding_server,
            storage_qdrant=storage_qdrant,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
        return instance

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        index_root: str = ".rag_index",
        device: str = "cuda",
        verbose: int = 1,
        embedding_server: Optional[EmbeddingServerConfig] = None,
    ):
        """Load an index and the associated model from disk.

        Parameters:
            index_path (Union[str, Path]): Path to the index.
            device (str): The device to load the model on. Default is "cuda".
            embedding_server (Optional[EmbeddingServerConfig]): If set, embeddings are computed
                on the remote vLLM server instead of locally.

        Returns:
            cls (MultiModalRetrieverModel): Initialised instance with index loaded.
        """
        instance = cls()
        index_path = Path(index_path)
        instance.model = ColPaliModel.from_index(
            index_path,
            index_root=index_root,
            device=device,
            verbose=verbose,
            embedding_server=embedding_server,
        )
        return instance

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[int] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[
            Union[
                Dict[Union[str, int], Dict[str, Union[str, int]]],
                List[Dict[str, Union[str, int]]],
            ]
        ] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        description: str = "",
        ai_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Build an index from input documents.

        Parameters:
            input_path (Union[str, Path]): Path to the input documents.
            index_name (Optional[str]): The name of the index that will be built.
            doc_ids (Optional[List[Union[str, int]]]): List of document IDs.
            store_collection_with_index (bool): Whether to store the collection with the index.
            overwrite (bool): Whether to overwrite an existing index with the same name.
            metadata (Optional[...]): Per-document metadata dicts.
            description (str): Optional human-readable description of the corpus.
                If empty and ``ai_cfg`` is provided, auto-generated from per-document
                AI metadata ``short_description`` fields after indexing.
            ai_cfg (Optional[Dict]): Provider config for the summary LLM (same format as
                ``ai_metadata_provider_factory``).  Only used for auto-generation when
                ``description`` is empty.

        Returns:
            None
        """
        return self.model.index(
            input_path,
            index_name,
            doc_ids,
            store_collection_with_index,
            overwrite=overwrite,
            metadata=metadata,
            max_image_width=max_image_width,
            max_image_height=max_image_height,
            description=description,
            ai_cfg=ai_cfg,
            **kwargs,
        )

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Optional[int] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        """Add an item to an existing index.

        Parameters:
            input_item (Union[str, Path, Image.Image]): The item to add to the index.
            store_collection_with_index (bool): Whether to store the collection with the index.
            doc_id (Union[str, int]): The document ID for the item being added.
            metadata (Optional[Dict[str, Union[str, int]]]): Metadata for the document being added.

        Returns:
            None
        """
        return self.model.add_to_index(
            input_item, store_collection_with_index, doc_id, metadata=metadata
        )

    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Union[Dict[str, Any], MetadataFilter]] = None,
        return_base64_results: Optional[bool] = None,
    ) -> List[Result]:
        """Query an index.

        Parameters:
            query (Union[str, List[str]]): The query or queries to search for.
            k (int): The number of results to return. Default is 10.
            filter_metadata (Optional[Union[Dict[str, Any], MetadataFilter]]): Metadata to filter results by.
            return_base64_results (Optional[bool]): Whether to return base64-encoded image results.

        Returns:
            Union[List[Result], List[List[Result]]]: A list of Result objects or a list of lists of Result objects.
        """
        return self.model.search(query, k, filter_metadata, return_base64_results)

    def update_index_from_folder(
        self,
        folder: Union[str, Path],
        store_collection_with_index: bool = False,
        metadata_provider: Optional[Callable] = None,
        batch_size: int = 1,
        reindex_modified: bool = False,
    ) -> Dict[int, str]:
        return self.model.update_index_from_folder(
            folder,
            store_collection_with_index,
            metadata_provider,
            batch_size,
            reindex_modified,
        )

    @property
    def index_description(self) -> str:
        """Human-readable description of the indexed corpus (empty string if not set)."""
        return getattr(self.model, "index_description", "")

    def fetch(self, result: Result) -> Result:
        """Fetch a result from the index."""
        return self.model.fetch_result_img(result)

    def get_doc_ids_to_file_names(self):
        return self.model.get_doc_ids_to_file_names()

    def as_langchain_retriever(self, **kwargs: Any):
        return FORetrievalLangChain(model=self, kwargs=kwargs)
