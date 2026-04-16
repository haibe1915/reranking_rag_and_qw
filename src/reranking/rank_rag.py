from typing import List, Dict, Optional
import re

import numpy as np
from sentence_transformers import CrossEncoder

from src.rewriting.prompt_templates import RERANKING_PROMPT

class RankReranker:
    def __init__(self, 
                 method: str = "cross-encoder",
                 cross_encoder_model: str = "BAAI/bge-reranker-base",
                 llm_client=None):
        self.method = method
        self.llm_client = llm_client
        
        if method == "cross-encoder":
            self.cross_encoder = CrossEncoder(cross_encoder_model, device="cpu")
        else:
            self.cross_encoder = None
    
    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        ce_scores = [self.cross_encoder.predict([(query, d["text"][:300])])[0] for d in docs]
        llm_scores = [self.llm_score(query, d["text"]) for d in docs]

        def normalize(x):
            mn, mx = min(x), max(x)
            return [(i - mn) / (mx - mn + 1e-8) for i in x]

        ce_scores = normalize(ce_scores)
        llm_scores = normalize(llm_scores)

        for i, d in enumerate(docs):
            d["score"] = self.ce_weight * ce_scores[i] + self.llm_weight * llm_scores[i]

        return sorted(docs, key=lambda x: x["score"], reverse=True)
    
    def llm_score(self, query: str, doc: str) -> float:
        if not self.llm_client:
            return 0.0

        prompt = RERANKING_PROMPT.format(question=query, document=doc[:300])
        try:
            output = self.llm_client.generate(prompt, max_tokens=16, temperature=0.0)
            score = float(output.strip())
            return max(0.0, min(score / 10.0, 1.0))
        except Exception:
            return 0.0
    
    def _cross_encoder_rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        if not documents:
            return []

        # Filter URL documents trước khi rerank
        valid_docs = [d for d in documents if not d.get("text", "").startswith("http")]
        if not valid_docs:
            print("  All retrieved docs are URLs - check your index!")
            return documents[:top_k]

        pairs = [[query, doc["text"]] for doc in valid_docs]
        raw_scores = self.cross_encoder.predict(pairs)

        if raw_scores.ndim == 2:
            scores = raw_scores[:, 1]
        else:
            scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # Sanity check: nếu tất cả scores gần 0.5, warn
        score_std = np.std(scores)
        if score_std < 0.01:
            print(f"  All reranker scores ≈ {scores.mean():.3f} (std={score_std:.4f})")
            print("   Possible cause: documents are not meaningful text")

        for doc, score in zip(valid_docs, scores):
            doc["reranker_score"] = float(score)

        return sorted(valid_docs, key=lambda x: x["reranker_score"], reverse=True)[:top_k]
    
    def _llm_rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """LLM-based reranking"""
        if not self.llm_client or not documents:
            return documents[:top_k]
        
        scored: List[Dict] = []
        for doc in documents:
            prompt = RERANKING_PROMPT.format(
                question=query, document=doc["text"][:300]
            )
            try:
                response = self.llm_client.generate(prompt, [], max_tokens=10)
                score = self._parse_score(response)
            except Exception:
                score = 0.0
            doc = dict(doc)
            doc["reranker_score"] = score
            scored.append(doc)

        return sorted(scored, key=lambda d: d["reranker_score"], reverse=True)[:top_k]
    
    @staticmethod
    def _parse_score(text: str) -> float:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            val = float(nums[0])
            return min(val / 10.0, 1.0)
        return 0.0


class HybridReranker:
    """Combine cross-encoder and LLM reranking via weighted score fusion."""
    def __init__(
        self,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        llm_client=None,
        cross_encoder_weight: float = 0.7,
        llm_weight: float = 0.3,
    ):
        self.ce_reranker = RankReranker(
            method="cross-encoder", cross_encoder_model=cross_encoder_model
        )
        self.llm_reranker = RankReranker(method="llm", llm_client=llm_client)
        self.ce_weight = cross_encoder_weight
        self.llm_weight = llm_weight

    def rerank(
        self, query: str, documents: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        # Ensure each doc has a stable id
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc["id"] = i

        n = len(documents)

        # Cross-encoder scores
        ce_docs = self.ce_reranker.rerank(query, [dict(d) for d in documents], n)
        ce_map: Dict[int, float] = {d["id"]: d["reranker_score"] for d in ce_docs}

        # LLM scores (optional)
        if self.llm_reranker.llm_client:
            llm_docs = self.llm_reranker.rerank(
                query, [dict(d) for d in documents], n
            )
            llm_map: Dict[int, float] = {d["id"]: d["reranker_score"] for d in llm_docs}
        else:
            llm_map = {}

        # Fuse
        for doc in documents:
            ce_score = ce_map.get(doc["id"], 0.0)
            llm_score = llm_map.get(doc["id"], 0.0)
            if llm_map:
                doc["final_score"] = self.ce_weight * ce_score + self.llm_weight * llm_score
            else:
                doc["final_score"] = ce_score

        return sorted(documents, key=lambda d: d["final_score"], reverse=True)[:top_k]
