import os
from src.rewriting.query_rewriter import QueryRewriter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.rank_rag import RankReranker
from src.generation.llm_generator import LLMGenerator
from evaluation.evaluator import RAGEvaluator

def run_rag_pipeline(user_query, ground_truth=None):
    
    rewriter = QueryRewriter(method="multi-query-hyde")
    retriever = HybridRetriever(vector_db_path="data/vector_store")
    reranker = RankReranker(method="cross-encoder")
    generator = LLMGenerator(model_name="gpt-4")
    evaluator = RAGEvaluator()

    print(f"--- Đang xử lý truy vấn: {user_query} ---")

    expanded_queries = rewriter.rewrite(user_query) 
    
    raw_docs = retriever.retrieve(expanded_queries, top_k=10)

    ranked_docs = reranker.rerank(user_query, raw_docs)

    final_context = ranked_docs[:5] 

    answer = generator.generate(user_query, final_context)

    print(f"Câu trả lời: {answer}")

    if ground_truth:
        metrics = evaluator.calculate_all(
            query=user_query,
            answer=answer,
            context=final_context,
            ground_truth=ground_truth
        )
        return answer, metrics

    return answer

if __name__ == "__main__":
    # Ví dụ chạy thử nghiệm với câu hỏi tiếng Việt [3]
    sample_query = "đồ mặc mùa đông"
    sample_ground_truth = "Quần áo giữ ấm như áo khoác, hoodie, quần dài..."
    
    result, scores = run_rag_pipeline(sample_query, sample_ground_truth)
    
    print("\n--- Kết quả đánh giá Baseline/Cải tiến ---")
    print(f"EM Score: {scores['em']}") # Exact Match [11]
    print(f"F1 Score: {scores['f1']}") # F1 Score [11]
    print(f"RAGAS Faithfulness: {scores['faithfulness']}") # Khung RAGAS [10]