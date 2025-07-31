from typing import Optional

from pydantic import BaseModel


class Result(BaseModel):
    doc_id: str
    """The unique identifier for the document."""
    page_num: int
    """The page number within the document."""
    score: float
    """The relevance score of the document."""
    metadata: Optional[dict] = None
    """Additional metadata associated with the document."""
    base64: Optional[str] = None
    """Base64 encoded content of the document."""

    def dict(self):
        return {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "score": self.score,
            "metadata": self.metadata,
            "base64": self.base64,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()
