import numpy as np
from typing import List, Dict, Optional
import torch
from sentence_transformers import CrossEncoder
from src.rewriting.prompt_templates import RERANKING_PROMPT
import logging

logger = logging.getLogger(__name__)


class RankReranker:
    def __init__(self,
                 method: str = "cross-encoder",
                 cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                 llm_client=None,
                 ce_weight: float = 1.0,
                 llm_weight: float = 0.0):
        self.method = method
        self.llm_client = llm_client
        self.ce_weight = ce_weight
        self.llm_weight = llm_weight

        if method == "cross-encoder":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(cross_encoder_model, device=device, trust_remote_code=True)
        else:
            self.cross_encoder = None

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []

        if self.method == "cross-encoder":
            if self.cross_encoder is None:
                logger.warning("Cross-encoder not initialized, returning top-k docs")
                return docs[:top_k]
            pairs = [(query, d["text"]) for d in docs]
            raw_scores = self.cross_encoder.predict(pairs)
            ce_scores = list(1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=float).flatten())))
            llm_scores = [0.0] * len(docs)
        else:
            ce_scores = [0.0] * len(docs)
            llm_scores = [self.llm_score(query, d["text"]) for d in docs]

        if self.llm_client and self.llm_weight > 0:
            llm_scores = [self.llm_score(query, d["text"]) for d in docs]
        else:
            llm_scores = [0.0] * len(docs)

        def normalize(x):
            if not x:
                return []
            mn, mx = min(x), max(x)
            if mx == mn:
                return [0.5] * len(x)
            return [(i - mn) / (mx - mn + 1e-8) for i in x]

        norm_ce = normalize(ce_scores)
        norm_llm = normalize(llm_scores)

        for i, d in enumerate(docs):
            d["reranker_score"] = self.ce_weight * norm_ce[i] + self.llm_weight * norm_llm[i]
            d["score"] = d["reranker_score"]

        return sorted(docs, key=lambda x: x["reranker_score"], reverse=True)[:top_k]

    def llm_score(self, query: str, doc: str) -> float:
        if not self.llm_client:
            return 0.0

        doc_context = doc[:300]
        prompt = RERANKING_PROMPT.format(question=query, document=doc_context)

        try:
            output = self.llm_client.generate(prompt, max_tokens=16, temperature=0.0)
            score_str = output.strip()

            match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_str)
            if not match:
                logger.warning(f"LLM output is not a number: {score_str}")
                return 0.0
            score = float(match.group(1))
            return max(0.0, min(score / 10.0, 1.0))
        except Exception as e:
            logger.error(f"Error in LLM scoring: {e}")
            return 0.0

class ContextSelector:
    def __init__(self,
                 top_k: int = 3,
                 deduplicate: bool = True,
                 similarity_threshold: float = 0.85,
                 max_tokens: Optional[int] = None):
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens

    def select(self, ranked_docs: List[Dict]) -> List[Dict]:
        if not ranked_docs:
            return []

        if not self.deduplicate:
            selected = ranked_docs[:self.top_k]
        else:
            selected = self._deduplicate(ranked_docs)

        if self.max_tokens is not None:
            selected = self._trim_to_token_budget(selected)

        return selected

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _deduplicate(self, docs: List[Dict]) -> List[Dict]:
        selected = []
        for doc in docs:
            if len(selected) >= self.top_k:
                break
            is_duplicate = any(
                self._jaccard_similarity(doc["text"], s["text"]) >= self.similarity_threshold
                for s in selected
            )
            if not is_duplicate:
                selected.append(doc)
        return selected

    def _trim_to_token_budget(self, docs: List[Dict]) -> List[Dict]:
        result = []
        total = 0
        for doc in docs:
            approx_tokens = len(doc["text"].split())
            if total + approx_tokens > self.max_tokens:
                break
            result.append(doc)
            total += approx_tokens
        return result