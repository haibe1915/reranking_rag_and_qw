from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import re

class HybridRetriever:
    def __init__(self, vector_model="BAAI/bge-small-en-v1.5",
                 bm25_weight=0.5, vector_weight=0.5):
        self.vector_model  = SentenceTransformer(vector_model)
        self.bm25_weight   = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = None; self.corpus = []; self.faiss_index = None

    def build_index(self, documents: List[str]):
        valid_docs = [
          d for d in documents
          if d
          and isinstance(d, str)
          and not d.startswith("http")
          and len(d.strip()) > 20
      	]
    
        if not valid_docs:
            raise ValueError("No valid documents to index! Check your data fields.")

        if len(valid_docs) < len(documents):
            print(f"⚠️  Filtered {len(documents) - len(valid_docs)} invalid docs (URLs/empty)")
    
        self.corpus = valid_docs
        tokenized  = [self._tok(d) for d in documents]
        self.bm25  = BM25Okapi(tokenized)
        embs = self.vector_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        embs = normalize(embs, axis=1).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(embs.shape[1])
        self.faiss_index.add(embs)

    def retrieve(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        if not self.corpus: return []
        all_scores: Dict[int, float] = {}
        for q in queries:
            bm25_scores = np.array(self.bm25.get_scores(self._tok(q)))
            q_emb = self.vector_model.encode([q], convert_to_numpy=True)
            q_emb = normalize(q_emb, axis=1).astype('float32')
            dists, idxs = self.faiss_index.search(q_emb, min(top_k*2, len(self.corpus)))
            vec_scores = np.zeros(len(self.corpus))
            for rank, idx in enumerate(idxs[0]):
                vec_scores[idx] = 1/(1+dists[0][rank])
            def norm(arr):
                mn, mx = arr.min(), arr.max()
                return (arr-mn)/(mx-mn+1e-8)
            combined = self.bm25_weight*norm(bm25_scores) + self.vector_weight*norm(vec_scores)
            for doc_id, score in enumerate(combined):
                all_scores[doc_id] = all_scores.get(doc_id,0) + score
        top = sorted(all_scores.items(), key=lambda x:x[1], reverse=True)[:top_k]
        return [{"text": self.corpus[i], "score": float(s), "id": i} for i,s in top]

    def _tok(self, text):
        return re.sub(r'\s+',' ', text.lower()).strip().split()
