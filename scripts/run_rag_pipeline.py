import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import config
from src.data.data_loader import DataLoader
from src.generation.llm_generator import LLMGenerator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.rank_rag import RankReranker
from src.rewriting.query_rewriter import QueryRewriter
from evaluation.evaluator import RAGEvaluator

class RAGPipeline:
    def __init__(self, dataset_name: str, llm_provider: str = "ollama"):
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(config.data.data_root)
        self.llm = LLMGenerator(
            provider=llm_provider,
            ollama_model=config.llm.ollama_model,
            ollama_host=config.llm.ollama_host,
            groq_api_key=config.llm.groq_api_key,
            hf_model_name=config.llm.hf_model_name
        )
        self.rewriter = QueryRewriter(self.llm)
        self.retriever = HybridRetriever(
            vector_model=config.retriever.vector_model,
            bm25_weight=config.retriever.hybrid_bm25_weight,
            vector_weight=config.retriever.hybrid_vector_weight
        )
        self.reranker = RankReranker(
            method=config.reranker.reranker_type,
            cross_encoder_model=config.reranker.cross_encoder_model,
            llm_client=self.llm
        )
        self.evaluator = RAGEvaluator()
    
    def run(self, split: str = "test", max_samples: int = None):
        """Run pipeline on dataset"""
        print(f"\n🚀 Running RAG Pipeline on {self.dataset_name} ({split})")
        print("=" * 60)
        
        # Load data
        data = self.data_loader.load_dataset(self.dataset_name, split)
        if max_samples:
            data = data[:max_samples]
        
        # Build retriever index from all available documents
        all_documents = self._collect_all_documents()
        self.retriever.build_index(all_documents)
        
        # Process each sample
        results = []
        metrics_list = []
        
        for sample in tqdm(data, desc="Processing"):
            query = sample.get("question", sample.get("query", ""))
            ground_truth = sample.get("answer", sample.get("answer_text", ""))
            
            if not query or not ground_truth:
                continue
            
            # Run pipeline
            try:
                answer = self._run_single(query)
                metrics = self.evaluator.calculate_all(query, answer, [], ground_truth)
                
                results.append({
                    "query": query,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "metrics": metrics
                })
                
                metrics_list.append(metrics)
            except Exception as e:
                print(f"❌ Error processing: {e}")
                continue
        
        # Summary
        self._print_summary(results, metrics_list)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _run_single(self, query: str) -> str:
        """Run single query through pipeline"""
        # Rewrite
        queries = self.rewriter.rewrite(query, method="multi-query-hyde")
        
        # Retrieve
        docs = self.retriever.retrieve(queries, top_k=config.retriever.vector_top_k)
        
        # Rerank
        ranked_docs = self.reranker.rerank(query, docs, top_k=config.reranker.reranker_top_k)
        
        # Generate
        answer = self.llm.generate(query, ranked_docs, max_tokens=config.evaluation.context_max_length)
        
        return answer
    
    def _collect_all_documents(self) -> list:
        """Collect all documents from dataset"""
        train_data = self.data_loader.load_dataset(self.dataset_name, "train")
        
        documents = []
        for sample in train_data:
            # Extract context or passages
            if "context" in sample:
                documents.append(sample["context"])
            elif "passage" in sample:
                documents.append(sample["passage"])
            elif "text" in sample:
                documents.append(sample["text"])
        
        return documents if documents else ["Default document"]
    
    def _print_summary(self, results: list, metrics_list: list):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print(f"📊 Results: {len(results)} samples")
        
        if metrics_list:
            avg_metrics = {
                key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
                for key in metrics_list[0].keys()
            }
            
            print("\n📈 Average Metrics:")
            for key, value in avg_metrics.items():
                print(f"   {key.upper()}: {value:.4f}")
    
    def _save_results(self, results: list):
        """Save results to file"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{self.dataset_name}_results.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    parser.add_argument("--dataset", type=str, default="vhealthqa",
                       choices=["vhealthqa", "uit_viquad2", "vietnamese_rag", "vimpa", "all"])
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--llm_provider", type=str, default="ollama",
                       choices=["ollama", "groq", "huggingface"])
    
    args = parser.parse_args()
    
    datasets = (["vhealthqa", "uit_viquad2", "vietnamese_rag", "vimpa"] 
                if args.dataset == "all" 
                else [args.dataset])
    
    for dataset in datasets:
        pipeline = RAGPipeline(dataset, args.llm_provider)
        pipeline.run(args.split, args.max_samples)

if __name__ == "__main__":
    main()