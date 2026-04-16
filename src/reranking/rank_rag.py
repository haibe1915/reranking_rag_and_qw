import numpy as np
from typing import List, Dict, Optional
import re
import torch
from sentence_transformers import CrossEncoder
from src.rewriting.prompt_templates import RERANKING_PROMPT

class RankReranker:
    def __init__(self, 
                 method: str = "cross-encoder",
                 cross_encoder_model: str = "BAAI/bge-reranker-base",
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

        valid_docs = [d for d in docs if not d.get("text", "").startswith("http")]
        if not valid_docs:
            return docs[:top_k]

        if self.method == "cross-encoder" and self.cross_encoder:
            pairs = [[query, d["text"][:512]] for d in valid_docs]
            raw_scores = self.cross_encoder.predict(pairs)
            
            if isinstance(raw_scores, np.ndarray) and raw_scores.ndim == 2:
                ce_scores = raw_scores[:, 1]
            else:
                ce_scores = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            ce_scores = [0.0] * len(valid_docs)

        if self.llm_client and self.llm_weight > 0:
            llm_scores = [self.llm_score(query, d["text"]) for d in valid_docs]
        else:
            llm_scores = [0.0] * len(valid_docs)

        def normalize(x):
            if not x: return []
            mn, mx = min(x), max(x)
            if mx == mn: return [0.5] * len(x)
            return [(i - mn) / (mx - mn + 1e-8) for i in x]

        norm_ce = normalize(ce_scores)
        norm_llm = normalize(llm_scores)

        for i, d in enumerate(valid_docs):
            d["reranker_score"] = self.ce_weight * norm_ce[i] + self.llm_weight * norm_llm[i]
            d["score"] = d["reranker_score"]

        return sorted(valid_docs, key=lambda x: x["reranker_score"], reverse=True)[:top_k]
    
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