# hybrid_retriever_fixed.py
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import hashlib
import json
import re
import os
import pickle

# Vietnamese stopwords để BM25 không bị nhiễu
VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "có", "không", "được", "trong", "với", "cho",
    "các", "những", "này", "đó", "hay", "hoặc", "thể", "để", "một",
    "người", "bị", "tôi", "bạn", "họ", "khi", "từ", "đến", "về",
    "như", "vì", "nên", "cũng", "đã", "sẽ", "đang", "rất", "nhiều"
}


class HybridRetriever:
    def __init__(
        self,
        vector_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        bm25_weight=0.3,      # ← Giảm từ 0.6 xuống 0.3
        vector_weight=0.7,    # ← Tăng từ 0.4 lên 0.7
        cache_dir="cache"
    ):
        self.vector_model = SentenceTransformer(vector_model, device="cpu")
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.bm25 = None
        self.corpus = []
        self.faiss_index = None

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _corpus_hash(self, documents: List[str]) -> str:
        content = json.dumps(documents, ensure_ascii=False, sort_keys=False)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _tok(self, text: str) -> List[str]:
        """Tokenize với stopword filtering để BM25 chỉ match từ có nghĩa."""
        tokens = re.sub(r"\s+", " ", text.lower()).strip().split()
        # Lọc stopwords và token quá ngắn
        return [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]

    def build_index(self, documents: List[str]):
        valid_docs = [d for d in documents if isinstance(d, str) and d.strip()]

        emb_path    = os.path.join(self.cache_dir, "embeddings.npy")
        faiss_path  = os.path.join(self.cache_dir, "faiss.index")
        bm25_path   = os.path.join(self.cache_dir, "bm25.pkl")
        corpus_path = os.path.join(self.cache_dir, "corpus.pkl")
        hash_path   = os.path.join(self.cache_dir, "corpus_hash.txt")

        current_hash = self._corpus_hash(valid_docs)
        cache_valid = False
        if all(os.path.exists(p) for p in [emb_path, faiss_path, bm25_path, corpus_path, hash_path]):
            with open(hash_path, "r") as f:
                cached_hash = f.read().strip()
            cache_valid = (cached_hash == current_hash)

        if cache_valid:
            print("Loading cached index...")
            try:
                self.faiss_index = faiss.read_index(faiss_path)
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(corpus_path, "rb") as f:
                    self.corpus = pickle.load(f)
                return
            except Exception as e:
                # Cache bị corrupt (thường do FAISS version mismatch) → rebuild
                print(f"⚠️  Cache load failed ({e}), rebuilding index...")
                for p in [emb_path, faiss_path, bm25_path, corpus_path, hash_path]:
                    if os.path.exists(p):
                        os.remove(p)

        print("Building index from scratch...")
        self.corpus = valid_docs
        tokenized = [self._tok(d) for d in valid_docs]
        self.bm25 = BM25Okapi(tokenized)

        embs = self.vector_model.encode(
            valid_docs,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True
        )
        embs = normalize(embs, axis=1).astype("float32")
        dim = embs.shape[1]

        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embs)

        print("Saving cache...")
        np.save(emb_path, embs)
        faiss.write_index(self.faiss_index, faiss_path)
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)
        with open(hash_path, "w") as f:
            f.write(current_hash)
        print("Index built & cached")

    def retrieve(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        if not self.corpus or self.faiss_index is None:
            return []

        n = min(len(self.corpus), self.faiss_index.ntotal)  # clamp: không search quá số doc trong index

        # ── Dùng Reciprocal Rank Fusion thay vì average scores ──
        # RRF ổn định hơn khi nhiều queries có phân phối score khác nhau
        rrf_scores = np.zeros(n, dtype="float64")
        k_rrf = 60  # hằng số RRF chuẩn

        for q in queries:
            bm25_scores = np.array(self.bm25.get_scores(self._tok(q)), dtype="float64")

            q_emb = self.vector_model.encode([q], convert_to_numpy=True)
            q_emb = normalize(q_emb, axis=1).astype("float32")
            sims, idxs = self.faiss_index.search(q_emb, n)

            vec_scores = np.zeros(n, dtype="float64")
            for sim, idx in zip(sims[0], idxs[0]):
                vec_scores[idx] = float(sim)

            combined = (self.bm25_weight * self._norm(bm25_scores) +
                        self.vector_weight * self._norm(vec_scores))

            # Chuyển combined score → rank → RRF contribution
            ranks = np.argsort(np.argsort(-combined))  # rank từ 0
            rrf_scores += 1.0 / (k_rrf + ranks + 1)

        top_indices = np.argpartition(rrf_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(rrf_scores[top_indices])[::-1]]

        return [
            {"text": self.corpus[i], "score": float(rrf_scores[i]), "id": i}
            for i in top_indices
        ]

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn + 1e-8)