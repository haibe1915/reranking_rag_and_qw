from typing import List, Any

class HybridRetriever:
    """Minimal HybridRetriever stub used by main.py.

    Methods
    -------
    retrieve(queries: list, top_k: int) -> list
        Return a list of document dicts (stub returns simple placeholder docs).
    """

    def __init__(self, vector_db_path: str = None) -> None:
        self.vector_db_path = vector_db_path

    def retrieve(self, queries: List[Any], top_k: int = 10):
        # Return placeholder documents
        return [{"id": i, "text": f"document {i}", "score": 1.0} for i in range(top_k)]
