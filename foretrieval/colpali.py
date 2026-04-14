import base64
import importlib.util
import io
import logging
import os
from pathlib import Path
import shutil
import srsly
import tempfile
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any, Callable

from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2_5, ColQwen2_5_Processor, ColQwen2Processor
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import BitsAndBytesConfig

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        MultiVectorComparator,
        MultiVectorConfig,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from .docling_ingest import chunk_pdf_to_images
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

from .file_to_pdf import _convert_to_pdf
from .models_metadata import DocMetadata, MetadataFilter
from .objects import Result
from .plot_utils import draw_circle_on_max_patch, pil_from_base64, pil_to_base64_png, compute_patch_heatmap, majority_token_id, build_heatmap_overlays_base64
from .utils import _value_match

VERSION = "0.0.1"

# set the name for logging
logger = logging.getLogger(__name__)


# ColPaliModel supports two storage backends:
# - local backend: embeddings and mappings are stored in local files
# - Qdrant backend: embeddings are stored in Qdrant, while local sidecar files
#   are still used for metadata, image caches, and heatmap-related tensors
#
# The class is organized into:
# 1. initialization and loading
# 2. persistence
# 3. ingestion and indexing
# 4. search
# 5. result enrichment
# 6. file helpers

class ColPaliModel:
    # ============================================================
    # Initialization and index loading
    # ============================================================

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        index_root: str = ".foretrieval",
        device: Optional[Union[str, torch.device]] = None,
        ingestion: Dict[str, Any] = {"backend": "default"},
        storage_qdrant: bool = True,
        **kwargs,
    ):
        if isinstance(pretrained_model_name_or_path, Path):
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if ("colpali" not in pretrained_model_name_or_path.lower() and "colqwen2" not in pretrained_model_name_or_path.lower()):
            raise ValueError("This pre-release version of Byaldi only supports ColPali and ColQwen2 for now. Incorrect model name specified.")
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_name = self.pretrained_model_name_or_path
        self.verbose = verbose
        self.load_from_index = load_from_index
        self.index_root = index_root
        self.index_name = index_name
        self.kwargs = kwargs
        self.storage_qdrant = storage_qdrant

        self.ingestion = ingestion
        self.ingestion_backend = (self.ingestion.get("backend") or "default").lower()
        if self.ingestion_backend == "docling":
            if not _DOCLING_AVAILABLE:
                raise RuntimeError(
                    "The 'docling' ingestion backend requires the docling package.\n"
                    "Install it with:  pip install \"foretrieval[docling]\"\n"
                    "or:               uv add foretrieval --extra docling"
                )
            self.docling_cfg = self.ingestion.get("docling_cfg", {})

        self.n_gpu = torch.cuda.device_count() if n_gpu == -1 else n_gpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.collection = {}
        self.embed_id_to_extra = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()

        # local backend only
        self.indexed_embeddings = []
        self.embed_id_to_doc_id = {}

        self.enable_heatmaps = False
        self.enable_circle = False
        self.SOURCE_EXTS = {".doc", ".docx", ".rtf", ".odt", ".ppt", ".pptx", ".odp", ".xls", ".xlsx", ".ods", ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".epub", ".html"}
        self.IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

        self.qdrant_client = None
        self.qdrant_collection = index_name
        self.qdrant_path = None

        load_in_4bit = bool(kwargs.pop("load_in_4bit", False))
        load_in_8bit = bool(kwargs.pop("load_in_8bit", False))
        bnb_4bit_quant_type = str(kwargs.pop("bnb_4bit_quant_type", "nf4"))
        bnb_4bit_compute_dtype = str(kwargs.pop("bnb_4bit_compute_dtype", "float16"))

        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one quantization mode can be enabled: 4-bit or 8-bit.")

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError(
                    "Quantization requested but `bitsandbytes` is not installed. "
                    "Install it with `pip install bitsandbytes`."
                )
            if load_in_4bit:
                compute_dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                if bnb_4bit_compute_dtype not in compute_dtype_map:
                    raise ValueError(
                        "Invalid bnb_4bit_compute_dtype. Expected one of: "
                        "'float16', 'bfloat16', 'float32'."
                    )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype_map[bnb_4bit_compute_dtype],
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._quantization_config = quantization_config

        if self.storage_qdrant and self.index_name is not None:
            self.qdrant_path = Path(self.index_root) / self.index_name / "qdrant"
            self.qdrant_client = QdrantClient(path=str(self.qdrant_path))

        self.docling_dir = None
        if self.index_name is not None and self.ingestion_backend == "docling":
            self.docling_dir = Path(index_root) / self.index_name / "docling_chunks"
            self.docling_dir.mkdir(parents=True, exist_ok=True)

        self._load_model_and_processor()

        if not load_from_index:
            self.full_document_collection = False
            self.resize_stored_images = False
            self.max_image_width = None
            self.max_image_height = None
            self.highest_doc_id = -1
        else:
            self._load_index_state()

    def _load_model_and_processor(self):
        token = self.kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN")
        is_cuda = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        device_map = "cuda" if is_cuda else None

        if "colpali" in self.pretrained_model_name_or_path.lower():
            model_cls = ColPali
            processor_cls = ColPaliProcessor
        elif "colqwen2.5" in self.pretrained_model_name_or_path.lower():
            model_cls = ColQwen2_5
            processor_cls = ColQwen2_5_Processor
        else:
            model_cls = ColQwen2
            processor_cls = ColQwen2Processor

        self.model = model_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=self._quantization_config,
            token=token,
        )
        self.processor = processor_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            token=token,
        )

        self.model = self.model.eval()
        if device_map is None and not (self._load_in_4bit or self._load_in_8bit):
            self.model = self.model.to(self.device)

    def _load_index_state(self):
        if self.index_name is None:
            raise ValueError("No index name specified. Cannot load from index.")

        if self.storage_qdrant:
            self._load_qdrant_state()
        else:
            self._load_local_index()

    def _load_local_sidecars(self, index_path: Path):
        extra_path = index_path / "embed_id_to_extra.pt"
        if extra_path.exists():
            self.embed_id_to_extra = torch.load(extra_path, map_location="cpu")
            self.embed_id_to_extra = {int(k): v for k, v in self.embed_id_to_extra.items()}
        else:
            self.embed_id_to_extra = {}

        doc_names_path = index_path / "doc_ids_to_file_names.json.gz"
        if doc_names_path.exists():
            self.doc_ids_to_file_names = srsly.read_gzip_json(doc_names_path)
            self.doc_ids_to_file_names = {int(k): v for k, v in self.doc_ids_to_file_names.items()}
        else:
            self.doc_ids_to_file_names = {}

        metadata_path = index_path / "metadata.json.gz"
        if metadata_path.exists():
            self.doc_id_to_metadata = srsly.read_gzip_json(metadata_path)
            self.doc_id_to_metadata = {int(k): v for k, v in self.doc_id_to_metadata.items()}
        else:
            self.doc_id_to_metadata = {}
            
    def _load_local_index(self):
        index_path = Path(self.index_root) / self.index_name

        index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")
        self.full_document_collection = index_config.get("full_document_collection", False)
        self.resize_stored_images = index_config.get("resize_stored_images", False)
        self.max_image_width = index_config.get("max_image_width", None)
        self.max_image_height = index_config.get("max_image_height", None)

        if self.full_document_collection:
            collection_path = index_path / "collection"
            json_files = sorted(
                collection_path.glob("*.json.gz"),
                key=lambda x: int(x.stem.split(".")[0]),
            )
            for json_file in json_files:
                loaded_data = srsly.read_gzip_json(json_file)
                self.collection.update({int(k): v for k, v in loaded_data.items()})

        embeddings_path = index_path / "embeddings"
        embedding_files = sorted(
            embeddings_path.glob("embeddings_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        self.indexed_embeddings = []
        for file in embedding_files:
            self.indexed_embeddings.extend(torch.load(file, map_location="cpu"))

        self.embed_id_to_doc_id = srsly.read_gzip_json(index_path / "embed_id_to_doc_id.json.gz")
        self.embed_id_to_doc_id = {int(k): v for k, v in self.embed_id_to_doc_id.items()}

        self._load_local_sidecars(index_path)

    def _load_qdrant_state(self):
        self._ensure_qdrant_client()
        if self.qdrant_client is None:
            raise ValueError("Qdrant client is not initialized.")

        assert self.index_name is not None
        self.qdrant_collection = self.index_name

        if not self.qdrant_client.collection_exists(self.qdrant_collection):
            raise ValueError(
                f"Qdrant collection '{self.qdrant_collection}' does not exist."
            )

        index_path = Path(self.index_root) / self.index_name

        index_config_path = index_path / "index_config.json.gz"
        if index_config_path.exists():
            index_config = srsly.read_gzip_json(index_config_path)
            self.full_document_collection = index_config.get("full_document_collection", False)
            self.resize_stored_images = index_config.get("resize_stored_images", False)
            self.max_image_width = index_config.get("max_image_width", None)
            self.max_image_height = index_config.get("max_image_height", None)
        else:
            self.full_document_collection = False
            self.resize_stored_images = False
            self.max_image_width = None
            self.max_image_height = None

        self._load_local_sidecars(index_path)

        self.highest_doc_id = max(self.doc_id_to_metadata.keys(), default=-1)
        self.doc_ids = set(self.doc_id_to_metadata.keys())

    def _ensure_qdrant_client(self):
        if not self.storage_qdrant:
            return

        if not _QDRANT_AVAILABLE:
            raise RuntimeError(
                "The Qdrant storage backend requires the qdrant-client package.\n"
                "Install it with:  pip install \"foretrieval[qdrant]\"\n"
                "or:               uv add foretrieval --extra qdrant"
            )

        if self.index_name is None:
            raise ValueError("index_name must be set before initializing Qdrant.")

        if self.qdrant_client is None:
            self.qdrant_path = Path(self.index_root) / self.index_name / "qdrant"
            self.qdrant_client = QdrantClient(path=str(self.qdrant_path))
            
    def _ensure_qdrant_collection(self, dim: Optional[int] = None):
        if not self.storage_qdrant:
            return

        self._ensure_qdrant_client()
        
        if self.qdrant_client is None:
            raise ValueError("Qdrant client is not initialized.")

        if self.index_name is None:
            raise ValueError("index_name must be set before creating Qdrant collection.")

        self.qdrant_collection = self.index_name

        if self.qdrant_client.collection_exists(self.qdrant_collection):
            return

        if dim is None:
            raise ValueError("Qdrant collection does not exist yet and no vector dimension was provided.")

        self._create_qdrant_collection(self.qdrant_collection, dim)

    def _create_qdrant_collection(self, collection_name: str, dim: int):
        if self.qdrant_client is None:
            raise ValueError("Qdrant client is not initialized.")

        if self.qdrant_client.collection_exists(collection_name):
            return

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    def _make_point_id(self, doc_id: int, page_id: int, chunk_id: Optional[int] = None) -> int:
        chunk_val = 0 if chunk_id is None else int(chunk_id)
        return int(doc_id) * 10_000_000 + int(page_id) * 10_000 + chunk_val

    def set_enable_heatmaps_and_circle(self, enable_heatmaps: bool, enable_circle: bool):
        self.enable_heatmaps = enable_heatmaps
        self.enable_circle = enable_circle

    # ============================================================
    # Persistence and index export
    # ============================================================

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        ingestion: Dict[str, Any] = {"backend": "default"},
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".foretrieval",
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ingestion=ingestion,
            n_gpu=n_gpu,
            verbose=verbose,
            load_from_index=False,
            index_root=index_root,
            device=device,
            **kwargs,
        )

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".foretrieval",
        **kwargs,
    ):
        index_path = Path(os.path.join(Path(index_root), Path(index_path)))
        index_config: dict = srsly.read_gzip_json(index_path / "index_config.json.gz")
        storage_backend = index_config.get("storage_backend", "local")
        storage_qdrant = storage_backend == "qdrant"

        instance = cls(
            pretrained_model_name_or_path=index_config["model_name"],
            n_gpu=n_gpu,
            index_name=index_path.name,
            verbose=verbose,
            load_from_index=True,
            index_root=str(index_path.parent),
            device=device,
            storage_qdrant=storage_qdrant,
            **kwargs,
        )

        return instance

    def _export_index(self):
        if self.index_name is None:
            raise ValueError("No index name specified. Cannot export.")

        index_path = Path(self.index_root) / self.index_name
        index_path.mkdir(parents=True, exist_ok=True)

        index_config = {
            "model_name": self.model_name,
            "full_document_collection": self.full_document_collection,
            "highest_doc_id": self.highest_doc_id,
            "resize_stored_images": (
                True if self.max_image_width and self.max_image_height else False
            ),
            "max_image_width": self.max_image_width,
            "max_image_height": self.max_image_height,
            "library_version": VERSION,
            "storage_backend": "qdrant" if self.storage_qdrant else "local",
            "qdrant_collection": self.qdrant_collection if self.storage_qdrant else None,
        }
        srsly.write_gzip_json(index_path / "index_config.json.gz", index_config)

        # local-only sidecars still useful in both modes
        torch.save(self.embed_id_to_extra, index_path / "embed_id_to_extra.pt")
        srsly.write_gzip_json(index_path / "doc_ids_to_file_names.json.gz", self.doc_ids_to_file_names)
        srsly.write_gzip_json(index_path / "metadata.json.gz", self.doc_id_to_metadata)

        if self.full_document_collection:
            collection_path = index_path / "collection"
            collection_path.mkdir(exist_ok=True)
            for i in range(0, len(self.collection), 500):
                chunk = dict(list(self.collection.items())[i : i + 500])
                srsly.write_gzip_json(collection_path / f"{i}.json.gz", chunk)

        if not self.storage_qdrant:
            embeddings_path = index_path / "embeddings"
            embeddings_path.mkdir(exist_ok=True)
            num_embeddings = len(self.indexed_embeddings)
            chunk_size = 500
            for i in range(0, num_embeddings, chunk_size):
                chunk = self.indexed_embeddings[i : i + chunk_size]
                torch.save(chunk, embeddings_path / f"embeddings_{i}.pt")

            srsly.write_gzip_json(
                index_path / "embed_id_to_doc_id.json.gz",
                self.embed_id_to_doc_id,
            )

        if self.verbose > 0:
            print(f"Index exported to {index_path}")

    # ============================================================
    # Index building and ingestion
    # ============================================================

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[List[int]] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[Union[List[DocMetadata], Dict[int, DocMetadata]]] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        batch_size: int = 1,
    ) -> Union[Dict[int, str], None]:
        if (
            self.index_name is not None
            and (index_name is None or self.index_name == index_name)
            and not overwrite
        ):
            raise ValueError(
                f"An index named {self.index_name} is already loaded.",
                "Use add_to_index() to add to it or search() to query it.",
                "Pass a new index_name to create a new index.",
                "Exiting indexing without doing anything...",
            )
        if index_name is None:
            raise ValueError("index_name must be specified to create a new index.")

        index_path = Path(os.path.join(Path(self.index_root), Path(index_name)))
        if index_path.exists():
            if not overwrite and (
                (index_path.is_dir() and len(list(index_path.iterdir())) > 0)
                or index_path.is_file()
            ):
                logger.warning(
                    f"An index named {index_name} already exists.",
                    "Use overwrite=True to delete the existing index and build a new one.",
                    "Exiting indexing without doing anything...",
                )
                return None
            else:
                logger.info(
                    f"overwrite is on. Deleting existing index {index_name} to build a new one."
                )
                shutil.rmtree(index_path)

        if store_collection_with_index:
            self.full_document_collection = True
        self.index_name = index_name
        if self.storage_qdrant:
            self.qdrant_collection = self.index_name
            self._ensure_qdrant_client()

        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

        input_path = Path(input_path)
        if not hasattr(self, "highest_doc_id") or overwrite is True:
            self.highest_doc_id = -1

        if input_path.is_dir():
            # Sort by filename so the ordering is deterministic and consistent
            # with build_metadata_list_for_dir(), which also sorts by p.name.
            # Without this, two independent iterdir() calls on the same
            # directory can return different orderings, silently misaligning
            # a metadata list with the wrong documents.
            items = sorted(input_path.iterdir(), key=lambda p: p.name)
            if doc_ids is not None and len(doc_ids) != len(items):
                raise ValueError(
                    f"Number of doc_ids ({len(doc_ids)}) does not match number of documents ({len(items)})"
                )
            if metadata is not None and len(metadata) != len(items):
                raise ValueError(
                    f"Number of metadata entries ({len(metadata)}) does not match number of documents ({len(items)})"
                )
            for i, item in tqdm(
                enumerate(items), total=len(items), desc="Indexing files"
            ):
                doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1
                if metadata is None:
                    doc_md = None
                elif isinstance(metadata, list):
                    doc_md = metadata[i]  # align list on items
                elif isinstance(metadata, dict):
                    doc_md = metadata.get(doc_id)
                else:
                    doc_md = metadata[doc_id] if metadata else None

                try:
                    self.add_to_index(
                        item,
                        store_collection_with_index,
                        doc_id=doc_id,
                        metadata=doc_md,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    logger.warning(f"Skipping faulty PDF {item}:\n{str(e)}")
                    continue

        else:
            if metadata is not None and len(metadata) != 1:
                raise ValueError(
                    "For a single document, metadata should be a list with one dictionary"
                )
            doc_id = doc_ids[0] if doc_ids else self.highest_doc_id + 1
            doc_metadata = metadata[0] if metadata else None
            self.add_to_index(
                input_path,
                store_collection_with_index,
                doc_id=doc_id,
                metadata=doc_metadata,
            )
            # self.doc_ids_to_file_names[doc_id] = str(input_path)

        self._export_index()
        if self.highest_doc_id == -1:
            logger.warning("No documents were indexed.")

        return self.doc_ids_to_file_names

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        store_collection_with_index: bool,
        doc_id: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Union[List[DocMetadata], DocMetadata]] = None,
        batch_size: int = 1,
    ) -> Dict[int, str]:
        if self.index_name is None:
            raise ValueError(
                "No index loaded. Use index() to create or load an index first."
            )
        if not hasattr(self, "highest_doc_id"):
            self.highest_doc_id = -1
        # Convert single inputs to lists for uniform processing
        if isinstance(input_item, (str, Path)) and Path(input_item).is_dir():
            input_items = list(Path(input_item).iterdir())
        else:
            input_items = (
                [input_item] if not isinstance(input_item, list) else input_item
            )

        doc_ids = (
            [doc_id]
            if isinstance(doc_id, int)
            else (doc_id if doc_id is not None else None)
        )

        # Validate input lengths
        if doc_ids and len(doc_ids) != len(input_items):
            raise ValueError(
                f"Number of doc_ids ({len(doc_ids)}) does not match number of input items ({len(input_items)})"
            )

        # Process each input item
        for i, item in enumerate(input_items):
            current_doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1 + i
            current_metadata = metadata if metadata else None

            if current_doc_id in self.doc_ids:
                raise ValueError(
                    f"Document ID {current_doc_id} already exists in the index"
                )

            self.highest_doc_id = max(self.highest_doc_id, current_doc_id)

            if isinstance(item, (str, Path)):
                item_path = Path(item)
                if item_path.is_dir():
                    self._process_directory(
                        item_path,
                        store_collection_with_index,
                        current_doc_id,
                        current_metadata,
                        batch_size,
                    )
                else:
                    stored_path = self._process_and_add_to_index(
                        item_path,
                        store_collection_with_index,
                        current_doc_id,
                        current_metadata,
                        batch_size,
                    )
                     # store the path for later use
                    if stored_path is None:
                        self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
                    else:
                        self.doc_ids_to_file_names[current_doc_id] = str(stored_path)

            elif isinstance(item, Image.Image):
                self._process_and_add_to_index(
                    item, store_collection_with_index, current_doc_id, current_metadata
                )
                self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

        self._export_index()
        return self.doc_ids_to_file_names

    def _process_directory(
        self,
        directory: Path,
        store_collection_with_index: bool,
        base_doc_id: int,
        metadata: Optional[Dict[str, Union[str, int]]],
        batch_size: int,
    ):
        for i, item in enumerate(directory.iterdir()):
            print(f"Indexing file: {item}")
            current_doc_id = base_doc_id + i
            stored_path = self._process_and_add_to_index(
                item, store_collection_with_index, current_doc_id, metadata, batch_size
            )
            if stored_path is None:
                self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
            else:
                self.doc_ids_to_file_names[current_doc_id] = str(stored_path)

    def _process_and_add_to_index(
        self,
        item: Union[Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        metadata: Optional[Dict[str, Union[str, int]]] = None,
        batch_size: int = 1,
    ) -> Optional[Path]:
        """
        Process and index an image or any file (converted to PDF if needed).
        Returns the 'canonical' path (PDF or image) used, or None for in-memory images.
        """
        if isinstance(item, Path):
            ext = item.suffix.lower()

            # 0) docling chunking (if enabled)
            if self.ingestion_backend == "docling":

                # 0) Convert to PDF if not already a PDF (or a PDF mirror)
                if ext == ".pdf":
                    pdf_file = item.resolve()
                else:
                    existing_pdf = self._find_existing_pdf(item)
                    if existing_pdf is not None:
                        pdf_file = existing_pdf
                    else:
                        # 2) otherwise convert it
                        pdf_file = _convert_to_pdf(item)
                        if pdf_file is None:
                            logger.warning(f"Docling ingestion: failed to convert {item} to PDF. Skipping.")
                            return None  # skip

                # 1) Sauvegarder les chunks Docling sur disque
                if self.docling_dir is None:
                    assert self.index_name is not None, "index_name must be set to use docling ingestion"
                    self.docling_dir = Path(self.index_root) / self.index_name / "docling_chunks"
                    self.docling_dir.mkdir(parents=True, exist_ok=True)
                chunks = chunk_pdf_to_images(pdf_file, output_dir=self.docling_dir)

                # 2) indexer
                for i in range(0, len(chunks), batch_size):
                    batch_chunks, batch_page_ids, batch_chunk_ids = [], [], []
                    for j in range(i, min(i + batch_size, len(chunks))):
                        ch = chunks[j]
                        image = Image.open(ch.path)
                        batch_chunks.append(image)
                        batch_page_ids.append(ch.page_id)
                        batch_chunk_ids.append(ch.elem_id)
                    self._add_to_index(
                        batch_chunks,
                        store_collection_with_index,
                        doc_id,
                        page_ids=batch_page_ids,
                        chunk_ids=batch_chunk_ids,
                        metadata=metadata,
                    )

                return Path(pdf_file).resolve()
            # if docling fails => fallback to default below

            # 1) images disque : pas docling
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(item)
                self._add_to_index(image, store_collection_with_index, doc_id, metadata=metadata)
                return item.resolve()  # <--- disk image path

            # 2) default (code existant) : pdf / convert_to_pdf / pdf2image
            elif ext == ".pdf":
                pdf_file = item.resolve()

                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(
                        pdf_file,
                        thread_count=os.cpu_count() - 1,
                        output_folder=path,
                        paths_only=True,
                    )
                    for i in range(0, len(images), batch_size):
                        batch_images, batch_page_ids = [], []
                        for j in range(i, min(i + batch_size, len(images))):
                            image_path = images[j]
                            image = Image.open(image_path)
                            batch_images.append(image)
                            batch_page_ids.append(j + 1)
                        self._add_to_index(
                            batch_images,
                            store_collection_with_index,
                            doc_id,
                            page_ids=batch_page_ids,
                            metadata=metadata,
                        )
                return pdf_file
            # Si c'est une image, on l'indexe directement sans conversion (ne passe pas par Docling)
            else:
                # 1) if a valid twin PDF already exists → use it
                existing_pdf = self._find_existing_pdf(item)
                if existing_pdf is not None:
                    pdf_file = existing_pdf
                else:
                    # 2) otherwise convert it
                    pdf_file = _convert_to_pdf(item)
                    if pdf_file is None:
                        return None  # skip

                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(
                        pdf_file,
                        thread_count=os.cpu_count() - 1,
                        output_folder=path,
                        paths_only=True,
                    )
                    for i in range(0, len(images), batch_size):
                        batch_images, batch_page_ids = [], []
                        for j in range(i, min(i + batch_size, len(images))):
                            image_path = images[j]
                            image = Image.open(image_path)
                            batch_images.append(image)
                            batch_page_ids.append(j + 1)
                        self._add_to_index(
                            batch_images,
                            store_collection_with_index,
                            doc_id,
                            page_ids=batch_page_ids,
                            metadata=metadata,
                        )
                return Path(pdf_file).resolve()

        elif isinstance(item, Image.Image):
            self._add_to_index(item, store_collection_with_index, doc_id, metadata=metadata)
            return None  # in-memory
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

    def _add_to_index(
        self,
        images: Union[Image.Image, List[Image.Image]],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        page_ids: Union[int, List[int]] = 1,
        chunk_ids: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        # Convert single image to list for uniform processing
        if isinstance(images, Image.Image):
            images = [images]

        # Convert single page_id to list for uniform processing
        if isinstance(page_ids, int):
            page_ids = [page_ids]

        # Convert chunk_ids to list if needed
        if chunk_ids is None:
            chunk_ids = [None] * len(images)
        elif isinstance(chunk_ids, int):
            chunk_ids = [chunk_ids]

        # Validate input lengths
        if len(images) != len(page_ids):
            raise ValueError(f"Number of images ({len(images)}) does not match number of page_ids ({len(page_ids)})")
        if len(images) != len(chunk_ids):
            raise ValueError(f"Number of images ({len(images)}) does not match number of chunk_ids ({len(chunk_ids)})")

        # Check for existing entries (by chunk if exist (docling) or page)
        if not self.storage_qdrant:
            if any(c is not None for c in chunk_ids):
                for chunk_id in chunk_ids:
                    if chunk_id is None:
                        continue
                    if any(
                        entry["doc_id"] == doc_id and entry.get("chunk_id") == chunk_id
                        for entry in self.embed_id_to_doc_id.values()
                    ):
                        raise ValueError(f"Document ID {doc_id} with chunk ID {chunk_id} already exists in the index")
            else:
                for page_id in page_ids:
                    if any(
                        entry["doc_id"] == doc_id and entry["page_id"] == page_id
                        for entry in self.embed_id_to_doc_id.values()
                    ):
                        raise ValueError(f"Document ID {doc_id} with page ID {page_id} already exists in the index")

        # Process images in batches
        processed_images = self.processor.process_images(images)

        # --- NEW: garder des infos CPU pour debug/heatmap ---
        # (before moving to GPU)
        input_ids_cpu = processed_images["input_ids"].detach().cpu()
        grid_cpu = processed_images["image_grid_thw"].detach().cpu()
        # Optionnel mais utile : taille originale de l'image
        orig_sizes = [img.size for img in images]  # (W,H) PIL

        # Generate embeddings
        with torch.inference_mode():
            processed_images = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in processed_images.items()
            }
            embeddings = self.model(**processed_images)

        # 1. Compute embeddings
        # 2. Ensure backend storage and check duplicates
        # 3. Store embeddings and sidecar data
        embeddings_list = list(torch.unbind(embeddings.to("cpu")))

        if self.storage_qdrant:
            dim = int(embeddings_list[0].shape[-1])
            self._ensure_qdrant_collection(dim=dim)

        for i, (embedding, page_id, chunk_id) in enumerate(zip(embeddings_list, page_ids, chunk_ids)):
            if self.storage_qdrant:
                if self.qdrant_client is None or self.qdrant_collection is None:
                    raise ValueError("Qdrant is not initialized correctly.")

                # --- check existence ---
                point_id = self._make_point_id(
                    int(doc_id),
                    int(page_id),
                    int(chunk_id) if chunk_id is not None else None,
                )
                found = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[point_id],
                    with_payload=False,
                    with_vectors=False,
                )
                if found:
                    if chunk_id is not None:
                        raise ValueError(f"Document ID {doc_id} with chunk ID {chunk_id} already exists")
                    raise ValueError(f"Document ID {doc_id} with page ID {page_id} already exists")

                # --- insertion ---
                payload = {
                    "doc_id": int(doc_id),
                    "page_id": int(page_id),
                    "chunk_id": int(chunk_id) if chunk_id is not None else None,
                    "metadata": (
                        metadata.as_jsonable() if isinstance(metadata, DocMetadata)
                        else (DocMetadata(**metadata).as_jsonable() if isinstance(metadata, dict) else {})
                    ),
                }

                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding.float().numpy().tolist(),
                            payload=payload,
                        )
                    ],
                )

                self.embed_id_to_extra[point_id] = {
                    "input_ids": input_ids_cpu[i],
                    "image_grid_thw": grid_cpu[i],
                    "orig_size": orig_sizes[i],
                }

                if store_collection_with_index:
                    img_str = self._post_process_image(images[i])
                    self.collection[int(point_id)] = img_str

            else:   
                entry = {
                    "doc_id": int(doc_id),
                    "page_id": int(page_id),
                }
                if chunk_id is not None:
                    entry["chunk_id"] = int(chunk_id)
                embed_id = len(self.indexed_embeddings)
                self.indexed_embeddings.append(embedding)
                self.embed_id_to_doc_id[embed_id] = entry

                self.embed_id_to_extra[embed_id] = {
                    "input_ids": input_ids_cpu[i],
                    "image_grid_thw": grid_cpu[i],
                    "orig_size": orig_sizes[i],
                }

                if store_collection_with_index:
                    img_str = self._post_process_image(images[i])
                    self.collection[int(embed_id)] = img_str

    # ============================================================
    # Index maintenance
    # ============================================================

    def update_index_from_folder(
        self,
        folder: Union[str, Path],
        store_collection_with_index: bool = False,
        metadata_provider: Optional[Callable] = None,
        batch_size: int = 1,
        reindex_modified: bool = False,
    ) -> Dict[int, str]:
        """
        Adds only NEW files from a folder to the current index.
        - Ignores mirror PDFs if a source file with the same stem exists.
        - Avoids duplicates by checking if the file (or its sibling PDF) is already indexed.
        - If reindex_modified=True, also reindexes files whose modification date has changed
        (compared to the stored canonical target). In this case, the old doc_id is removed
        and the file is reindexed with a new doc_id.

        metadata_provider: callable(Path) -> Optional[Dict[str, Union[str,int]]]
        """
        folder = Path(folder)
        assert folder.is_dir(), f"{folder} n'est pas un dossier existant."

        # set of already known paths
        known = self._already_indexed_paths()

        # inverse map to find a doc_id from a canonical path
        inverse_map: Dict[str, int] = {}
        for did, p in self.doc_ids_to_file_names.items():
            if p and p != "In-memory Image":
                try:
                    inverse_map[str(Path(p).resolve())] = int(did)
                except Exception:
                    inverse_map[p] = int(did)

        added = 0
        updated = 0

        for item in sorted(folder.iterdir()):
            if item.is_dir():
                # (optional) recursive descent if you want
                continue

            ext = item.suffix.lower()

            # 1) ignore mirror PDFs (if a source exists)
            if ext == ".pdf" and self._is_mirror_pdf(item):
                if self.verbose > 1:
                    print(f"[skip] Mirror PDF ignored: {item}")
                continue

            # 2) avoid duplicates: path itself OR its sibling PDF already known?
            cand_keys = self._candidate_keys(item)
            if any(k in known for k in cand_keys) and not reindex_modified:
                if self.verbose > 1:
                    print(f"[skip] Already indexed: {item}")
                continue

            # 3) reindex_modified handling
            if reindex_modified:
                # if we find a known key, check mtime
                target_key = None
                for k in cand_keys:
                    if k in known:
                        target_key = k
                        break

                if target_key is not None:
                    try:
                        src_stat = item.stat().st_mtime
                        tgt_stat = Path(target_key).stat().st_mtime
                    except Exception:
                        src_stat, tgt_stat = None, None

                    # reindex only if the src is newer
                    if (
                        src_stat is not None
                        and tgt_stat is not None
                        and src_stat <= tgt_stat
                    ):
                        if self.verbose > 1:
                            print(f"[skip] Unchanged (mtime): {item}")
                        continue

                    # remove the old doc_id and (simply) replace it with a new one
                    old_doc_id = inverse_map.get(target_key)
                    if old_doc_id is not None:
                        if self.verbose > 0:
                            print(
                                f"[update] Reindexing (modified): {item} (doc_id {old_doc_id})"
                            )
                        # NB: no granular deletion API implemented;
                        # the simplest is to add a new doc_id and, if needed,
                        # mark the old one as obsolete via a metadata field or
                        # maintain a 'tombstones' list. Here we just add a new one.
                        updated += 1

            # 4) index this file (new or updated)
            doc_id = self.highest_doc_id + 1
            md = metadata_provider(item) if metadata_provider else None
            stored_path = self._process_and_add_to_index(
                item,
                store_collection_with_index=store_collection_with_index,
                doc_id=doc_id,
                metadata=md,
                batch_size=batch_size,
            )
            if stored_path is None:
                self.doc_ids_to_file_names[doc_id] = "In-memory Image"
            else:
                self.doc_ids_to_file_names[doc_id] = str(Path(stored_path).resolve())

            self.doc_ids.add(doc_id)
            self.highest_doc_id = max(self.highest_doc_id, doc_id)
            added += 1

        # export to persist mappings
        self._export_index()

        if self.verbose > 0:
            print(f"[incr] added: {added} | reindexed: {updated}")

        return self.doc_ids_to_file_names

    def remove_from_index(self):
        raise NotImplementedError("This method is not implemented yet.")

    # ============================================================
    # Search
    # ============================================================

    def _encode_search_query(self, query: str):
        with torch.inference_mode():
            batch_query = self.processor.process_queries([query])
            batch_query = {
                kk: vv.to(self.device).to(
                    self.model.dtype
                    if vv.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else vv.dtype
                )
                for kk, vv in batch_query.items()
            }
            embeddings_query = self.model(**batch_query)
            qs = list(torch.unbind(embeddings_query.to("cpu")))

        input_ids = batch_query["input_ids"][0].detach().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)
        valid_idxs = [i for i, tok in enumerate(tokens) if tok not in {"<|endoftext|>", "Query", ":"}]
        return [qs[0][valid_idxs]]

    def _search_local(
        self,
        qs,
        k: int,
        filter_metadata: Optional[Dict[str, str]],
        return_base64_results: bool,
    ) -> List[Result]:
        k = min(k, len(self.indexed_embeddings))

        if filter_metadata:
            req_embeddings, req_embedding_ids = self.filter_embeddings(filter_metadata)
            if not req_embeddings:
                logger.warning(
                    "Metadata filter matched no documents — returning empty results."
                )
                return []
        else:
            req_embeddings = self.indexed_embeddings
            req_embedding_ids = list(range(len(self.indexed_embeddings)))

        scores = self.processor.score(qs, req_embeddings, device=self.device).cpu().numpy()
        top_pages = scores.argsort(axis=1)[0][-k:][::-1].tolist()

        results: List[Result] = []
        for embed_id in top_pages:
            adjusted_embed_id = req_embedding_ids[embed_id]
            doc_info = self.embed_id_to_doc_id[adjusted_embed_id]

            result = Result(
                doc_id=doc_info["doc_id"],
                page_num=int(doc_info["page_id"]),
                chunk_num=int(doc_info["chunk_id"]) if doc_info.get("chunk_id") is not None else None,
                score=float(scores[0][embed_id]),
                metadata=self.doc_id_to_metadata.get(int(doc_info["doc_id"]), {}),
                base64=self.collection.get(adjusted_embed_id) if return_base64_results else None,
            )

            extra = self.embed_id_to_extra.get(adjusted_embed_id)
            if (self.enable_heatmaps or self.enable_circle) and extra is not None:
                result = self._attach_heatmaps_local(
                    result=result,
                    q_emb=qs[0],
                    p_emb=self.indexed_embeddings[adjusted_embed_id],
                    extra=extra,
                    k=k,
                )

            results.append(result)

        return self._finalize_results(results, return_base64_results)

    def _search_qdrant(
        self,
        qs,
        k: int,
        filter_metadata: Optional[Dict[str, str]],
        return_base64_results: bool,
    ) -> List[Result]:
        if self.qdrant_client is None:
            raise ValueError("Qdrant client is not initialized.")
        if self.qdrant_collection is None:
            raise ValueError("Qdrant collection is not set.")

        qfilter = self._build_qdrant_filter(filter_metadata)

        response = self.qdrant_client.query_points(
            collection_name=self.qdrant_collection,
            query=qs[0].float().numpy().tolist(),
            query_filter=qfilter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        points = response.points if hasattr(response, "points") else response

        results: List[Result] = []
        for point in points:
            payload = point.payload or {}
            doc_id = int(payload["doc_id"])
            page_id = int(payload["page_id"])
            chunk_id = payload.get("chunk_id")

            point_id = point.id

            result = Result(
                doc_id=doc_id,
                page_num=page_id,
                chunk_num=int(chunk_id) if chunk_id is not None else None,
                score=float(point.score),
                metadata=payload.get("metadata", self.doc_id_to_metadata.get(doc_id, {})),
                base64=self.collection.get(point_id) if return_base64_results else None,
            )

            extra = self.embed_id_to_extra.get(point_id)
            if (self.enable_heatmaps or self.enable_circle) and extra is not None:
                retrieved = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[point_id],
                    with_payload=False,
                    with_vectors=True,
                )
                if retrieved:
                    p_emb = torch.tensor(retrieved[0].vector)
                    result = self._attach_heatmaps_local(
                        result=result,
                        q_emb=qs[0],
                        p_emb=p_emb,
                        extra=extra,
                        k=k,
                    )

            results.append(result)

        return self._finalize_results(results, return_base64_results)

    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, str]] = None,
        return_base64_results: Optional[bool] = None
    ) -> List[Result]:

        if return_base64_results is None:
            return_base64_results = bool(self.collection)

        if k < 1:
            return []

        if not self.storage_qdrant:
            k = min(k, len(self.indexed_embeddings))

        qs = self._encode_search_query(query)

        if self.storage_qdrant:
            return self._search_qdrant(
                qs=qs,
                k=k,
                filter_metadata=filter_metadata,
                return_base64_results=return_base64_results,
            )

        return self._search_local(
            qs=qs,
            k=k,
            filter_metadata=filter_metadata,
            return_base64_results=return_base64_results,
        )

    def _build_qdrant_filter(self, filter_metadata: Optional[Dict[str, str]]):
        if not filter_metadata:
            return None

        must = []
        for key, value in filter_metadata.items():
            must.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=must)

    def filter_embeddings(self, filter_metadata: Union[Dict[str, Any], MetadataFilter]):
        # support dict -> Pydantic model
        f = (
            filter_metadata
            if isinstance(filter_metadata, MetadataFilter)
            else MetadataFilter(**filter_metadata)
        )

        req_doc_ids = []
        for did, md in self.doc_id_to_metadata.items():
            # md is stored as a JSONable dict
            if _value_match(md, f):
                req_doc_ids.append(int(did))

        req_doc_ids = list(set(req_doc_ids))

        req_embedding_ids = [
            eid
            for eid, doc in self.embed_id_to_doc_id.items()
            if int(doc["doc_id"]) in req_doc_ids
        ]
        req_embeddings = [
            ie
            for idx, ie in enumerate(self.indexed_embeddings)
            if idx in req_embedding_ids
        ]
        return req_embeddings, req_embedding_ids

    # ============================================================
    # Result enrichment and visualization
    # ============================================================

    def _get_image_token_id_from_extra(self, extra: dict) -> int:
        if hasattr(self.processor, "image_token_id"):
            return int(self.processor.image_token_id)
        return majority_token_id(extra["input_ids"])

    def _attach_heatmaps_local(self, result: Result, q_emb, p_emb, extra: dict, k: int) -> Result:
        img_tok = self._get_image_token_id_from_extra(extra)

        result.metadata = dict(result.metadata or {})
        heat_soft, heat_global = None, None

        if self.enable_circle or self.enable_heatmaps:
            heat_soft, _ = compute_patch_heatmap(
                q_emb=q_emb,
                p_emb=p_emb,
                input_ids=extra["input_ids"],
                image_grid_thw=extra["image_grid_thw"],
                image_token_id=img_tok,
                mode="soft_topk",
                topk=min(k, 8),
                temperature=0.2,
                normalize=False,
            )

        if self.enable_heatmaps:
            heat_global, _ = compute_patch_heatmap(
                q_emb=q_emb,
                p_emb=p_emb,
                input_ids=extra["input_ids"],
                image_grid_thw=extra["image_grid_thw"],
                image_token_id=img_tok,
                mode="global_sum",
                topk=k,
                temperature=0.2,
                normalize=False,
            )

        hm = {"soft_topk": {"heat_2d": heat_soft}}
        if self.enable_heatmaps:
            hm["global_sum"] = {"heat_2d": heat_global}
        result.metadata["heatmaps"] = hm

        return result

    def _finalize_results(self, results: List[Result], return_base64_results: bool) -> List[Result]:
        if not return_base64_results:
            return results

        for r in results:
            self.fetch_result_img(r)

        for r in results:
            if not r.base64:
                continue

            meta = r.metadata or {}
            need_overlay, need_circle = bool(self.enable_heatmaps), bool(self.enable_circle)

            if not (need_overlay or need_circle):
                continue

            hm = meta.get("heatmaps") or {}
            img = None
            if (need_overlay and hm) or need_circle:
                img = pil_from_base64(r.base64)

            if need_overlay and hm:
                meta["heatmap_overlays_base64"] = build_heatmap_overlays_base64(
                    img=img,
                    heatmaps=hm,
                    interps=("nearest", "bilinear"),
                    alpha=0.45,
                    cmap="jet",
                    shift_x=0.0,
                    shift_y=0.0,
                    patch_grow_pct=300.0,
                    grow_mode="mean",
                )

            if need_circle:
                soft = (hm.get("soft_topk") or {}).get("heat_2d")
                if soft is not None:
                    img_marked = draw_circle_on_max_patch(img=img, heat_2d=soft)
                    meta["soft_topk_max_patch_circle_base64"] = pil_to_base64_png(img_marked)

            r.metadata = meta

        return results

    def fetch_result_img(self, result: Result) -> Result:
        if result.base64:
            return result

        doc_id = result.doc_id
        file_name = self.doc_ids_to_file_names.get(doc_id)
        if not file_name or file_name == "In-memory Image":
            return result  # nothing to do

        path = Path(file_name)

        # --- NEW: DOCling chunk images (saved next to the index) ---
        if self.ingestion_backend == "docling":
            try:
                if self.docling_dir is None:
                    assert self.index_name is not None, "index_name must be set to use docling ingestion"
                    self.docling_dir = Path(self.index_root) / self.index_name / "docling_chunks"
                    self.docling_dir.mkdir(parents=True, exist_ok=True)
                assert result.chunk_num is not None, f"Result.chunk_num must be defined"
                path_chunk = Path(self.docling_dir) / f"{path.stem}_p{result.page_num}_{result.chunk_num}.png"
                assert path_chunk.exists(), f"Path {path_chunk} for chunk {result.chunk_num} does not exists"
                image = Image.open(path_chunk)
                result.base64 = self._post_process_image(image)
                return result
                # if not found: fall back to existing behaviour (pdf/image)
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"[fetch_result_img] Docling chunk fetch error: {e}")

        ext = path.suffix.lower()

        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(path)
                result.base64 = self._post_process_image(image)
                return result

            if ext != ".pdf":
                # BEFORE converting, check if a valid paired PDF already exists
                sibling_pdf = self._find_existing_pdf(path)
                if sibling_pdf is not None:
                    self.doc_ids_to_file_names[doc_id] = str(sibling_pdf)
                    path = sibling_pdf
                    ext = ".pdf"
                else:
                    pdf_path = _convert_to_pdf(path)
                    if pdf_path and pdf_path.exists():
                        self.doc_ids_to_file_names[doc_id] = str(pdf_path)
                        path = pdf_path
                        ext = ".pdf"
                    else:
                        if self.verbose > 0:
                            print(
                                f"[fetch_result_img] Impossible de convertir {path} en PDF."
                            )
                        return result

            # PDF: extract the requested page
            with tempfile.TemporaryDirectory() as tmpdir:
                images = convert_from_path(
                    str(path),
                    thread_count=os.cpu_count() - 1,
                    first_page=result.page_num,
                    last_page=result.page_num,
                    paths_only=True,
                    output_folder=tmpdir,
                )
                image = Image.open(images[0])
                result.base64 = self._post_process_image(image)
            return result

        except Exception as e:
            if self.verbose > 0:
                print(f"[fetch_result_img] Erreur: {e}")
            return result

    def _post_process_image(self, image: Image.Image) -> str:
        # Resize image while maintaining aspect ratio
        if self.max_image_width and self.max_image_height:
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            if img_width > self.max_image_width:
                new_width = self.max_image_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_width = img_width
                new_height = img_height
            if new_height > self.max_image_height:
                new_height = self.max_image_height
                new_width = int(new_height * aspect_ratio)
            if self.verbose > 2:
                print(
                    f"Resizing image to {new_width}x{new_height}",
                    f"(aspect ratio {aspect_ratio:.2f}, original size {img_width}x{img_height},"
                    f"compression {new_width / img_width * new_height / img_height:.2f})",
                )
            image = image.resize((new_width, new_height), Image.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    # ============================================================
    # File helpers
    # ============================================================

    def _looks_like_pdf(self, path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < 5:
                return False
            with open(path, "rb") as f:
                return f.read(5) == b"%PDF-"
        except Exception:
            return False

    def _find_existing_pdf(self, src: Path) -> Optional[Path]:
        cand = src.with_suffix(".pdf")
        if cand.exists() and self._looks_like_pdf(cand):
            return cand.resolve()
        return None

    def _already_indexed_paths(self) -> set:
        """All already indexed 'canonical' targets (normalized paths)."""
        vals = set()
        for p in self.doc_ids_to_file_names.values():
            if not p or p == "In-memory Image":
                continue
            try:
                vals.add(str(Path(p).resolve()))
            except Exception:
                vals.add(p)
        return vals

    def _candidate_keys(self, path: Path) -> List[str]:
        """
        Possible keys to check in the index to avoid duplicates:
        - the path itself (resolved)
        - its sibling PDF if it exists
        """
        keys = []
        try:
            keys.append(str(path.resolve()))
        except Exception:
            keys.append(str(path))

        sibling_pdf = path.with_suffix(".pdf")
        if sibling_pdf.exists() and self._looks_like_pdf(sibling_pdf):
            try:
                keys.append(str(sibling_pdf.resolve()))
            except Exception:
                keys.append(str(sibling_pdf))
        return keys

    def _is_mirror_pdf(self, path: Path) -> bool:
        """True if path is .pdf and source file with the same *stem* exists."""
        if path.suffix.lower() != ".pdf":
            return False
        stem = path.with_suffix("")
        parent = path.parent
        for ext in self.SOURCE_EXTS:
            if Path(os.path.join(parent, f"{stem.name}{ext}")).exists():
                return True
        return False
