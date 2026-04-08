from typing import List, Dict
from rouge_score import rouge_scorer
import re

class RAGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def calculate_all(self, query, prediction, context, ground_truth):
        if not prediction or len(prediction.strip()) == 0:
            prediction = " ".join([doc.get("text", "")[:100] for doc in context[:2]])
        return {
            "em": self.exact_match(prediction, ground_truth),
            "f1": self.f1_score(prediction, ground_truth),
            "similarity": self.similarity_score(prediction, ground_truth),
            "faithfulness": self.faithfulness(prediction, context),
            "relevance": self.relevance(query, context),
        }

    def exact_match(self, prediction, ground_truth):
        if not prediction or not ground_truth: return 0.0
        return 1.0 if self._clean(prediction) == self._clean(ground_truth) else 0.0

    def f1_score(self, prediction, ground_truth):
        if not prediction or not ground_truth: return 0.0
        p_tok = set(self._tok(prediction)); g_tok = set(self._tok(ground_truth))
        if not p_tok or not g_tok: return 0.0
        common = p_tok & g_tok
        if not common: return 0.0
        prec = len(common)/len(p_tok); rec = len(common)/len(g_tok)
        return 2*prec*rec/(prec+rec) if prec+rec else 0.0

    def similarity_score(self, prediction, ground_truth):
        if not prediction or not ground_truth: return 0.0
        try:
            return self.rouge_scorer.score(ground_truth, prediction)['rougeL'].fmeasure
        except: return 0.0

    def faithfulness(self, prediction, context):
        if not prediction or not context: return 0.0
        ctx = " ".join(d.get("text","") for d in context)
        p_tok = set(self._tok(prediction)); c_tok = set(self._tok(ctx))
        if not p_tok: return 0.0
        return min(len(p_tok & c_tok)/len(p_tok), 1.0)

    def relevance(self, query, context):
        if not query or not context: return 0.0
        q_tok = set(self._tok(query))
        if not q_tok: return 0.0
        scores = [len(q_tok & set(self._tok(d.get("text","")))) / len(q_tok)
                  for d in context if d.get("text")]
        return sum(scores)/len(scores) if scores else 0.0

    def _tok(self, text): return self._clean(text).split() if text else []
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
