from typing import Any, Dict

class RAGEvaluator:
    """Minimal RAGEvaluator stub used by main.py.

    Methods
    -------
    calculate_all(query, answer, context, ground_truth) -> dict
        Return basic evaluation metrics as a dict.
    """

    def __init__(self) -> None:
        pass

    def calculate_all(self, query: str, answer: str, context: Any, ground_truth: str) -> Dict[str, float]:
        # Very simple placeholder metrics
        em = 1.0 if answer.strip() == ground_truth.strip() else 0.0
        f1 = em  # placeholder: exact-match toggles f1 for now
        faithfulness = 1.0 if "[stub" not in answer else 0.5
        return {"em": em, "f1": f1, "faithfulness": faithfulness}
