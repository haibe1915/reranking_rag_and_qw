from typing import List, Any

class RankReranker:
    """Minimal RankReranker stub used by main.py.

    Methods
    -------
    rerank(query: str, docs: list) -> list
        Return reranked documents (stub returns docs unchanged).
    """

    def __init__(self, method: str = "cross-encoder") -> None:
        self.method = method

    def rerank(self, query: str, docs: List[Any]):
        # No-op stub: return docs as-is
        return docs
