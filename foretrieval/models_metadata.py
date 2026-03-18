from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, field_validator


class DocMetadata(BaseModel):
    source_path: Optional[str] = None
    stem: Optional[str] = None
    ext: Optional[str] = None
    mime: Optional[str] = None
    mtime: Optional[datetime] = None
    page_count: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    document_type: Optional[str] = None
    short_description: Optional[str] = None

    # --- Useful Normalisations ---
    @field_validator("ext", mode="before")
    def _norm_ext(cls, v):
        if v is None:
            return v
        v = str(v).strip().lower()
        return v if v.startswith(".") else f".{v}"

    @field_validator("language", "author", "title", mode="before")
    def _norm_str(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("tags", mode="before")
    def _norm_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(t).strip().lower() for t in v]

    @field_validator("mtime", mode="before")
    def _parse_dt(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        # support 'Z'
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))

    def as_jsonable(self) -> Dict[str, Any]:
        # useful for export (datetime -> isoformat)
        d = self.model_dump()
        if d.get("mtime"):
            d["mtime"] = d["mtime"].isoformat()
        return d


class MetadataFilter(BaseModel):
    # simple values OR lists
    language: Optional[Union[str, List[str]]] = None
    ext: Optional[Union[str, List[str]]] = None
    tags: Optional[Union[str, List[str]]] = None
    document_type: Optional[Union[str, List[str]]] = None
    # operators on mtime (ISO)
    mtime: Optional[Dict[str, str]] = None  # ex: {">=":"2025-09-01T00:00:00Z"}

    # global logic
    logic: str = "AND"  # "AND" or "OR"

    class Config:
        extra = "allow"  # allow other metadata keys if needed

    @field_validator("ext", mode="before")
    def _norm_filter_ext(cls, v):
        def nx(s):
            s = s.strip().lower()
            return s if s.startswith(".") else f".{s}"

        if v is None:
            return v
        if isinstance(v, list):
            return [nx(x) for x in v]
        return nx(v)

    @field_validator("tags", mode="before")
    def _norm_filter_tags(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = [v]
        return [str(t).strip().lower() for t in v]


def build_metadata_list_for_dir(
    input_dir: Path, provider: Callable[[Path], Dict[str, Any]]
) -> List[Optional[DocMetadata]]:
    """Build a metadata list for every entry in ``input_dir``.

    The list is sorted by filename (``p.name``) so that its order matches
    the sorted iteration used by ``ColPaliModel.index()`` when it receives a
    directory path.  Using an unsorted ``iterdir()`` in both places could
    produce a different ordering between two independent calls on the same
    directory, silently misaligning metadata with the wrong documents.

    Args:
        input_dir: Directory whose contents will be enumerated.
        provider: Callable that receives a ``Path`` and returns a metadata
            dict compatible with ``DocMetadata``.

    Returns:
        A list aligned with ``sorted(input_dir.iterdir(), key=lambda p: p.name)``.
        Files produce ``DocMetadata`` instances; sub-directories produce ``None``.
    """
    items = sorted(input_dir.iterdir(), key=lambda p: p.name)
    md_list: List[Optional[DocMetadata]] = []
    for p in items:
        if p.is_file():
            raw = provider(p)
            md_list.append(DocMetadata(**raw) if raw else None)
        else:
            md_list.append(None)
    return md_list
