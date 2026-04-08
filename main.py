import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import config
from src.generation.llm_generator import LLMGenerator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.rank_rag import RankReranker, HybridReranker
from src.rewriting.query_rewriter import QueryRewriter
from evaluation.evaluator import RAGEvaluator


def run_rag_pipeline(user_query: str, ground_truth: str = None, verbose: bool = True):
    """
    Run complete RAG pipeline
    
    Args:
        user_query: User's question
        ground_truth: Expected answer (for evaluation)
        verbose: Print intermediate results
    
    Returns:
        answer: Generated answer
        metrics: Evaluation metrics (if ground_truth provided)
    """
    
    if verbose:
        print(f"\n--- Processing: {user_query} ---")
    
    # Initialize components
    llm = LLMGenerator(provider=config.llm.provider)
    rewriter = QueryRewriter(llm)
    retriever = HybridRetriever()
    reranker = RankReranker(method=config.reranker.reranker_type, llm_client=llm)
    evaluator = RAGEvaluator()
    
    # Step 1: Query rewriting (Multi-query + HyDE + Code-switching)
    if config.query_rewriting_enabled:
        rewritten_queries = rewriter.rewrite(
            user_query, 
            method="multi-query-hyde"
        )
        if verbose:
            print(f"📝 Rewritten queries ({len(rewritten_queries)}):")
            for i, q in enumerate(rewritten_queries, 1):
                print(f"   {i}. {q}")
    else:
        rewritten_queries = [user_query]
    
    # Step 2: Retrieval (Hybrid: BM25 + Vector search)
    raw_docs = retriever.retrieve(rewritten_queries, top_k=config.retriever.vector_top_k)
    
    if verbose:
        print(f"\n📚 Retrieved {len(raw_docs)} documents")
        for i, doc in enumerate(raw_docs[:3], 1):
            print(f"   {i}. {doc['text'][:100]}... (score: {doc['score']:.3f})")
    
    # Step 3: Reranking (Cross-encoder or LLM)
    ranked_docs = reranker.rerank(user_query, raw_docs, top_k=config.reranker.reranker_top_k)
    
    if verbose:
        print(f"\n🔄 Reranked documents (top {len(ranked_docs)}):")
        for i, doc in enumerate(ranked_docs, 1):
            score = doc.get("reranker_score", doc.get("score", 0))
            print(f"   {i}. {doc['text'][:100]}... (score: {score:.3f})")
    
    # Step 4: Context selection (top-k documents)
    final_context = ranked_docs[:config.reranker.reranker_top_k]
    
    # Step 5: Generation (using Ollama/Groq/HuggingFace)
    answer = llm.generate(user_query, final_context, max_tokens=config.context_max_length)
    
    if verbose:
        print(f"\n💬 Answer: {answer}")
    
    # Step 6: Evaluation (if ground truth provided)
    metrics = None
    if ground_truth:
        metrics = evaluator.calculate_all(
            query=user_query,
            prediction=answer,
            context=final_context,
            ground_truth=ground_truth
        )
        
        if verbose:
            print(f"\n📊 Evaluation metrics:")
            print(f"   EM Score: {metrics.get('em', 0):.4f}")
            print(f"   F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"   Similarity: {metrics.get('similarity', 0):.4f}")
            print(f"   Faithfulness: {metrics.get('faithfulness', 0):.4f}")
            print(f"   Relevance: {metrics.get('relevance', 0):.4f}")
    
    return answer, metrics


def main():
    parser = argparse.ArgumentParser(description="Vietnamese RAG Pipeline")
    
    parser.add_argument("--query", type=str, default="Machine learning là gì?",
                       help="User query")
    parser.add_argument("--ground_truth", type=str, default=None,
                       help="Ground truth answer for evaluation")
    parser.add_argument("--llm_provider", type=str, default="ollama",
                       choices=["ollama", "groq", "huggingface"])
    parser.add_argument("--llm_model", type=str, default=None,
                       help="LLM model name")
    parser.add_argument("--reranker_type", type=str, default="cross-encoder",
                       choices=["cross-encoder", "llm", "none"])
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print intermediate results")
    
    args = parser.parse_args()
    
    # Update config from arguments
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.ollama_model = args.llm_model
    if args.reranker_type:
        config.reranker.reranker_type = args.reranker_type
    
    if args.interactive:
        # Interactive mode
        print("🤖 Vietnamese RAG Pipeline - Interactive Mode")
        print("=" * 50)
        print("Enter 'quit' to exit\n")
        
        while True:
            query = input("❓ Enter query: ").strip()
            if query.lower() == "quit":
                break
            
            ground_truth = input("📝 Ground truth (optional): ").strip() or None
            
            answer, metrics = run_rag_pipeline(query, ground_truth, verbose=True)
    else:
        # Single query mode
        print("🤖 Vietnamese RAG Pipeline")
        print("=" * 50)
        
        answer, metrics = run_rag_pipeline(
            args.query,
            args.ground_truth,
            verbose=args.verbose
        )
        
        if metrics is None and args.ground_truth:
            print(f"\n❌ Evaluation failed")


if __name__ == "__main__":
    main()