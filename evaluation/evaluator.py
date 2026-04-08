from typing import Any, Dict, Set
import re

class RAGEvaluator:
    """Minimal RAGEvaluator stub used by main.py.

    Methods
    -------
    calculate_all(query, answer, context, ground_truth) -> dict
        Return basic evaluation metrics as a dict.
    """

    def __init__(self) -> None:
        pass

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison: lowercase and remove punctuation/extra spaces."""
        text = text.lower()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_tokens(self, text: str) -> Set[str]:
        """Tokenize text into a set of words."""
        normalized = self._normalize_text(text)
        return set(normalized.split())

    def _calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score based on token-level overlap."""
        pred_tokens = self._get_tokens(prediction)
        truth_tokens = self._get_tokens(ground_truth)

        # Handle edge cases
        if len(pred_tokens) == 0 and len(truth_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0

        # Calculate precision, recall, and F1
        common = pred_tokens & truth_tokens
        num_same = len(common)

        precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = num_same / len(truth_tokens) if len(truth_tokens) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _calculate_em(self, prediction: str, ground_truth: str) -> float:
        """Calculate Exact Match score."""
        pred_normalized = self._normalize_text(prediction)
        truth_normalized = self._normalize_text(ground_truth)
        return 1.0 if pred_normalized == truth_normalized else 0.0

    def calculate_all(self, query: str, answer: str, context: Any, ground_truth: str) -> Dict[str, float]:
        """Calculate F1 and EM scores for the answer against ground truth."""
        em = self._calculate_em(answer, ground_truth)
        f1 = self._calculate_f1(answer, ground_truth)
        faithfulness = 1.0 if "[stub" not in answer else 0.5
        return {"em": em, "f1": f1, "faithfulness": faithfulness}
