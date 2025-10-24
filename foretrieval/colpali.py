import base64
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, cast, Any, Callable
from datetime import datetime

import srsly
import torch
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColQwen2Processor,
)
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from .models_metadata import DocMetadata, MetadataFilter
from .file_to_pdf import _convert_to_pdf
from .objects import Result

VERSION = "0.0.1"

# set the name for logging
logger = logging.getLogger(__name__)


class ColPaliModel:
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        index_root: str = ".foretrieval",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        if isinstance(pretrained_model_name_or_path, Path):
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if (
            "colpali" not in pretrained_model_name_or_path.lower()
            and "colqwen2" not in pretrained_model_name_or_path.lower()
        ):
            raise ValueError(
                "This pre-release version of Byaldi only supports ColPali and ColQwen2 for now. Incorrect model name specified."
            )

        if verbose > 0:
            print(
                f"Verbosity is set to {verbose} ({'active' if verbose == 1 else 'loud'}). Pass verbose=0 to make quieter."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_name = self.pretrained_model_name_or_path
        self.n_gpu = torch.cuda.device_count() if n_gpu == -1 else n_gpu
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.index_name = index_name
        self.verbose = verbose
        self.load_from_index = load_from_index
        self.index_root = index_root
        self.kwargs = kwargs
        self.collection = {}
        self.indexed_embeddings = []
        self.embed_id_to_doc_id = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()

        self.SOURCE_EXTS = {
            ".doc",
            ".docx",
            ".rtf",
            ".odt",
            ".ppt",
            ".pptx",
            ".odp",
            ".xls",
            ".xlsx",
            ".ods",
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".epub",
            ".html",
        }

        self.IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

        if "colpali" in pretrained_model_name_or_path.lower():
            self.model = ColPali.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=(
                    "cuda"
                    if device == "cuda"
                    or (isinstance(device, torch.device) and device.type == "cuda")
                    else None
                ),
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2_5.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=(
                    "cuda"
                    if device == "cuda"
                    or (isinstance(device, torch.device) and device.type == "cuda")
                    else None
                ),
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=(
                    "cuda"
                    if device == "cuda"
                    or (isinstance(device, torch.device) and device.type == "cuda")
                    else None
                ),
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        self.model = self.model.eval()

        if "colpali" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColPaliProcessor,
                ColPaliProcessor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColQwen2_5_Processor,
                ColQwen2_5_Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColQwen2Processor,
                ColQwen2Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )

        self.device = device
        # TODO: not sure, need to be verified
        # if device != "cuda" and not (
        #     isinstance(device, torch.device) and device.type == "cuda"
        # ):
        self.model = self.model.to(device)

        if not load_from_index:
            self.full_document_collection = False
            self.highest_doc_id = -1
        else:
            if self.index_name is None:
                raise ValueError("No index name specified. Cannot load from index.")

            index_path = Path(os.path.join(Path(index_root), Path(self.index_name)))
            index_config = srsly.read_gzip_json(
                os.path.join(index_path, "index_config.json.gz")
            )
            self.full_document_collection = index_config.get(
                "full_document_collection", False
            )
            self.resize_stored_images = index_config.get("resize_stored_images", False)
            self.max_image_width = index_config.get("max_image_width", None)
            self.max_image_height = index_config.get("max_image_height", None)

            if self.full_document_collection:
                collection_path = Path(os.path.join(index_path, "collection"))
                json_files = sorted(
                    collection_path.glob("*.json.gz"),
                    key=lambda x: int(x.stem.split(".")[0]),
                )

                for json_file in json_files:
                    loaded_data = srsly.read_gzip_json(json_file)
                    self.collection.update({int(k): v for k, v in loaded_data.items()})

                if self.verbose > 0:
                    print(
                        "You are using in-memory collection. This means every image is stored in memory."
                    )
                    print(
                        "You might want to rethink this if you have a large collection!"
                    )
                    print(
                        f"Loaded {len(self.collection)} images from {len(json_files)} JSON files."
                    )

            embeddings_path = Path(os.path.join(index_path, "embeddings"))
            embedding_files = sorted(
                embeddings_path.glob("embeddings_*.pt"),
                key=lambda x: int(x.stem.split("_")[1]),
            )
            self.indexed_embeddings = []
            for file in embedding_files:
                self.indexed_embeddings.extend(torch.load(file))

            self.embed_id_to_doc_id = srsly.read_gzip_json(
                os.path.join(index_path, "embed_id_to_doc_id.json.gz")
            )
            # Restore keys to integers
            self.embed_id_to_doc_id = {
                int(k): v for k, v in self.embed_id_to_doc_id.items()
            }
            self.highest_doc_id = max(
                int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
            )
            self.doc_ids = set(
                int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
            )
            try:
                # We don't want this error out with indexes created prior to 0.0.2
                self.doc_ids_to_file_names = srsly.read_gzip_json(
                    os.path.join(index_path, "doc_ids_to_file_names.json.gz")
                )
                self.doc_ids_to_file_names = {
                    int(k): v for k, v in self.doc_ids_to_file_names.items()
                }
            except FileNotFoundError:
                pass

            # Load metadata
            metadata_path = Path(os.path.join(index_path, "metadata.json.gz"))
            if metadata_path.exists():
                self.doc_id_to_metadata = srsly.read_gzip_json(metadata_path)
                # Convert metadata keys to integers
                self.doc_id_to_metadata = {
                    int(k): v for k, v in self.doc_id_to_metadata.items()
                }
            else:
                self.doc_id_to_metadata = {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".foretrieval",
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
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
        index_config = srsly.read_gzip_json(
            os.path.join(index_path, "index_config.json.gz")
        )

        instance = cls(
            pretrained_model_name_or_path=index_config["model_name"],
            n_gpu=n_gpu,
            index_name=index_path.name,
            verbose=verbose,
            load_from_index=True,
            index_root=str(index_path.parent),
            device=device,
            **kwargs,
        )

        return instance

    def _export_index(self):
        if self.index_name is None:
            raise ValueError("No index name specified. Cannot export.")

        index_path = Path(os.path.join(Path(self.index_root), Path(self.index_name)))
        index_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_path = Path(os.path.join(index_path, "embeddings"))
        embeddings_path.mkdir(exist_ok=True)
        num_embeddings = len(self.indexed_embeddings)
        chunk_size = 500
        for i in range(0, num_embeddings, chunk_size):
            chunk = self.indexed_embeddings[i : i + chunk_size]
            torch.save(chunk, os.path.join(embeddings_path, f"embeddings_{i}.pt"))

        # Save index config
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
        }
        srsly.write_gzip_json(
            Path(os.path.join(index_path, "index_config.json.gz")), index_config
        )

        # Save embed_id_to_doc_id mapping
        srsly.write_gzip_json(
            Path(os.path.join(index_path, "embed_id_to_doc_id.json.gz")),
            self.embed_id_to_doc_id,
        )

        # Save doc_ids_to_file_names
        srsly.write_gzip_json(
            Path(os.path.join(index_path, "doc_ids_to_file_names.json.gz")),
            self.doc_ids_to_file_names,
        )

        # Save metadata
        srsly.write_gzip_json(
            Path(os.path.join(index_path, "metadata.json.gz")), self.doc_id_to_metadata
        )

        # Save collection if using in-memory collection
        if self.full_document_collection:
            collection_path = Path(os.path.join(index_path, "collection"))
            collection_path.mkdir(exist_ok=True)
            for i in range(0, len(self.collection), 500):
                chunk = dict(list(self.collection.items())[i : i + 500])
                srsly.write_gzip_json(
                    Path(os.path.join(collection_path, f"{i}.json.gz")), chunk
                )

        if self.verbose > 0:
            print(f"Index exported to {index_path}")

    def _is_mirror_pdf(self, path: Path) -> bool:
        """True if path is .pdf and source file with the same *stem* exists."""
        if path.suffix.lower() != ".pdf":
            return False
        stem = path.with_suffix("")
        parent = path.parent
        for ext in self.SOURCE_EXTS:
            if (os.path.join(parent, f"{stem.name}{ext}")).exists():
                return True
        return False

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
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

        input_path = Path(input_path)
        if not hasattr(self, "highest_doc_id") or overwrite is True:
            self.highest_doc_id = -1

        if input_path.is_dir():
            items = list(input_path.iterdir())
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

                self.add_to_index(
                    item,
                    store_collection_with_index,
                    doc_id=doc_id,
                    metadata=doc_md,
                    batch_size=batch_size,
                )
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
                    # stocke le chemin réellement exploitable plus tard
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
            if ext == ".pdf":
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

            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(item)
                self._add_to_index(
                    image, store_collection_with_index, doc_id, metadata=metadata
                )
                return item.resolve()  # <--- disk image path

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
            self._add_to_index(
                item, store_collection_with_index, doc_id, metadata=metadata
            )
            return None  # in-memory
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

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

    def _add_to_index(
        self,
        images: Union[Image.Image, List[Image.Image]],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        page_ids: Union[int, List[int]] = 1,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        # Convert single image to list for uniform processing
        if isinstance(images, Image.Image):
            images = [images]

        # Convert single page_id to list for uniform processing
        if isinstance(page_ids, int):
            page_ids = [page_ids]

        # Validate input lengths
        if len(images) != len(page_ids):
            raise ValueError(
                f"Number of images ({len(images)}) does not match number of page_ids ({len(page_ids)})"
            )

        # Check for existing entries
        for page_id in page_ids:
            if any(
                entry["doc_id"] == doc_id and entry["page_id"] == page_id
                for entry in self.embed_id_to_doc_id.values()
            ):
                raise ValueError(
                    f"Document ID {doc_id} with page ID {page_id} already exists in the index"
                )

        # Process images in batches
        processed_images = self.processor.process_images(images)

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

        # Add to index
        embeddings_list = list(torch.unbind(embeddings.to("cpu")))
        for i, (embedding, page_id) in enumerate(zip(embeddings_list, page_ids)):
            embed_id = len(self.indexed_embeddings)
            self.indexed_embeddings.append(embedding)
            self.embed_id_to_doc_id[embed_id] = {
                "doc_id": doc_id,
                "page_id": int(page_id),
            }

            if store_collection_with_index:
                img_str = self._post_process_image(images[i])
                self.collection[int(embed_id)] = img_str

        # Update highest_doc_id
        self.highest_doc_id = max(
            self.highest_doc_id,
            int(doc_id) if isinstance(doc_id, int) else self.highest_doc_id,
        )

        # Add metadata
        if isinstance(metadata, DocMetadata):
            self.doc_id_to_metadata[int(doc_id)] = metadata.as_jsonable()
        elif isinstance(metadata, dict):
            # compat: si certains appelants envoient encore des dicts
            self.doc_id_to_metadata[int(doc_id)] = DocMetadata(**metadata).as_jsonable()

        if self.verbose > 0:
            print(f"Added {len(images)} pages of document {doc_id} to index.")

    def remove_from_index(self):
        raise NotImplementedError("This method is not implemented yet.")

    def _parse_iso(s: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    def _any_in(a: List[str], b: List[str]) -> bool:
        sa = {x.strip().lower() for x in a}
        sb = {x.strip().lower() for x in b}
        return len(sa & sb) > 0

    def _value_match(meta: Dict[str, Any], f: MetadataFilter) -> bool:
        checks = []

        # language
        if f.language is not None:
            mv = (meta.get("language") or "").strip().lower()
            if isinstance(f.language, list):
                checks.append(mv in [x.strip().lower() for x in f.language])
            else:
                checks.append(mv == f.language.strip().lower())

        # ext
        if f.ext is not None:
            mv = (meta.get("ext") or "").strip().lower()
            candidates = f.ext if isinstance(f.ext, list) else [f.ext]
            checks.append(mv in candidates)

        # document_type
        if f.document_type is not None:
            mv = (meta.get("document_type") or "").strip().lower()
            cands = (
                f.document_type
                if isinstance(f.document_type, list)
                else [f.document_type]
            )
            checks.append(mv in [x.strip().lower() for x in cands])

        # tags (contains / intersection)
        if f.tags is not None:
            mv = [str(t).strip().lower() for t in (meta.get("tags") or [])]
            cands = f.tags if isinstance(f.tags, list) else [f.tags]
            checks.append(_any_in(mv, [str(x).strip().lower() for x in cands]))

        # mtime (opérateurs)
        if f.mtime is not None:
            m = meta.get("mtime")
            mdt = _parse_iso(m) if isinstance(m, str) else None
            if mdt is None:
                checks.append(False)
            else:
                ok = True
                for op, rhs in f.mtime.items():
                    rdt = _parse_iso(rhs)
                    if rdt is None:
                        ok = False
                        break
                    if op == ">=" and not (mdt >= rdt):
                        ok = False
                    if op == "<=" and not (mdt <= rdt):
                        ok = False
                    if op == ">" and not (mdt > rdt):
                        ok = False
                    if op == "<" and not (mdt < rdt):
                        ok = False
                    if op == "==" and not (mdt == rdt):
                        ok = False
                    if not ok:
                        break
                checks.append(ok)

        # autres clés libres (MetadataFilter.extra='allow')
        for k, v in f.__dict__.items():
            if k in {"language", "ext", "tags", "document_type", "mtime", "logic"}:
                continue
            if v is None:
                continue
            mv = meta.get(k)
            if isinstance(v, list):
                checks.append(
                    str(mv).strip().lower() in [str(x).strip().lower() for x in v]
                )
            else:
                checks.append(str(mv).strip().lower() == str(v).strip().lower())

        if not checks:
            return True  # pas de filtre = passe

        return all(checks) if f.logic.upper() == "AND" else any(checks)

    def filter_embeddings(self, filter_metadata: Union[Dict[str, Any], MetadataFilter]):
        # support dict → modèle Pydantic
        f = (
            filter_metadata
            if isinstance(filter_metadata, MetadataFilter)
            else MetadataFilter(**filter_metadata)
        )

        req_doc_ids = []
        for did, md in self.doc_id_to_metadata.items():
            # md est stocké en dict JSONable -> parse léger
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

    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, str]] = None,
        return_base64_results: Optional[bool] = None,
    ) -> List[Result]:
        # Set default value for return_base64_results if not provided
        if return_base64_results is None:
            return_base64_results = bool(self.collection)

        # Ensure k is not larger than the number of indexed documents
        k = min(k, len(self.indexed_embeddings))

        # Process query
        with torch.inference_mode():
            batch_query = self.processor.process_queries([query])
            batch_query = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in batch_query.items()
            }
            embeddings_query = self.model(**batch_query)
            qs = list(torch.unbind(embeddings_query.to("cpu")))

        # Get embeddings to search against
        if filter_metadata:
            req_embeddings, req_embedding_ids = self.filter_embeddings(filter_metadata)
        else:
            req_embeddings = self.indexed_embeddings
            req_embedding_ids = list(range(len(self.indexed_embeddings)))

        # Compute scores (toujours)
        scores = (
            self.processor.score(qs, req_embeddings, device=self.device).cpu().numpy()
        )

        # Get top k relevant pages
        top_pages = scores.argsort(axis=1)[0][-k:][::-1].tolist()

        # Create Result objects
        results = []
        for embed_id in top_pages:
            adjusted_embed_id = req_embedding_ids[embed_id]
            doc_info = self.embed_id_to_doc_id[adjusted_embed_id]

            result = Result(
                doc_id=doc_info["doc_id"],
                page_num=int(doc_info["page_id"]),
                score=float(scores[0][embed_id]),
                metadata=self.doc_id_to_metadata.get(int(doc_info["doc_id"]), {}),
                base64=self.collection.get(adjusted_embed_id)
                if return_base64_results
                else None,
            )
            results.append(result)

        if return_base64_results:
            for result in results:
                self.fetch_result_img(result)

        return results

    def encode_image(
        self, input_data: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> torch.Tensor:
        """
        Compute embeddings for one or more images, PDFs, folders, or image files.

        Args:
            input_data (Union[str, Image.Image, List[Union[str, Image.Image]]]):
                A single image, PDF path, folder path, image file path, or a list of these.

        Returns:
            torch.Tensor: The computed embeddings for the input data.
        """
        if not isinstance(input_data, list):
            input_data = [input_data]

        images = []
        for item in input_data:
            if isinstance(item, Image.Image):
                images.append(item)
            elif isinstance(item, str):
                if os.path.isdir(item):
                    # Process folder
                    for file in os.listdir(item):
                        if file.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                        ):
                            images.append(Image.open(os.path.join(item, file)))
                elif item.lower().endswith(".pdf"):
                    # Process PDF
                    with tempfile.TemporaryDirectory() as path:
                        pdf_images = convert_from_path(
                            item, thread_count=os.cpu_count() - 1, output_folder=path
                        )
                        images.extend(pdf_images)
                elif item.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                ):
                    # Process image file
                    images.append(Image.open(item))
                else:
                    raise ValueError(f"Unsupported file type: {item}")
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

        with torch.inference_mode():
            batch = self.processor.process_images(images)
            batch = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in batch.items()
            }
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def encode_query(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute embeddings for one or more text queries.

        Args:
            query (Union[str, List[str]]):
                A single text query or a list of text queries.

        Returns:
            torch.Tensor: The computed embeddings for the input query/queries.
        """
        if isinstance(query, str):
            query = [query]

        with torch.inference_mode():
            batch = self.processor.process_queries(query)
            batch = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in batch.items()
            }
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def get_doc_ids_to_file_names(self):
        return self.doc_ids_to_file_names

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

    def fetch_result_img(self, result: Result) -> Result:
        if result.base64:
            return result

        doc_id = result.doc_id
        file_name = self.doc_ids_to_file_names.get(doc_id)
        if not file_name or file_name == "In-memory Image":
            return result  # rien à faire

        path = Path(file_name)
        ext = path.suffix.lower()

        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(path)
                result.base64 = self._post_process_image(image)
                return result

            if ext != ".pdf":
                # AVANT de convertir, on regarde s'il y a déjà un PDF jumeau valide
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

            # PDF : on extrait la page demandée
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
