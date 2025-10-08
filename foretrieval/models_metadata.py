# models_metadata.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

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

    # --- Normalisations utiles ---
    @validator("ext", pre=True)
    def _norm_ext(cls, v):
        if v is None:
            return v
        v = str(v).strip().lower()
        return v if v.startswith(".") else f".{v}"

    @validator("language", "author", "title", pre=True)
    def _norm_str(cls, v):
        return v.strip() if isinstance(v, str) else v

    @validator("tags", pre=True)
    def _norm_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(t).strip().lower() for t in v]

    @validator("mtime", pre=True)
    def _parse_dt(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        # supporte 'Z'
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))

    def as_jsonable(self) -> Dict[str, Any]:
        # utile pour l'export (datetime -> isoformat)
        d = self.dict()
        if d.get("mtime"):
            d["mtime"] = d["mtime"].isoformat()
        return d


class MetadataFilter(BaseModel):
    # valeurs simples OU listes
    language: Optional[Union[str, List[str]]] = None
    ext: Optional[Union[str, List[str]]] = None
    tags: Optional[Union[str, List[str]]] = None
    document_type: Optional[Union[str, List[str]]] = None
    # opérateurs sur mtime (ISO)
    mtime: Optional[Dict[str, str]] = None  # ex: {">=":"2025-09-01T00:00:00Z"}

    # logique globale
    logic: str = "AND"  # "AND" ou "OR"

    class Config:
        extra = "allow"  # autorise d'autres clés metadata si besoin

    @validator("ext", pre=True)
    def _norm_filter_ext(cls, v):
        def nx(s): 
            s = s.strip().lower()
            return s if s.startswith(".") else f".{s}"
        if v is None:
            return v
        if isinstance(v, list):
            return [nx(x) for x in v]
        return nx(v)

    @validator("tags", pre=True)
    def _norm_filter_tags(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = [v]
        return [str(t).strip().lower() for t in v]
