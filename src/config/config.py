from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class LLMConfig(BaseSettings):
    provider: str = "groq"
    ollama_model: str = "mistral"
    ollama_host: str = "http://localhost:11434"
    ollama_temperature: float = 0.7
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-8b-8192"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    hf_model_name: str = "Qwen/Qwen2-7B-Instruct"
    hf_quantization: str = "4bit"
    hf_device: str = "cpu"

class RetrieverConfig(BaseSettings):
    retriever_type: str = "hybrid"  # bm25, vector, hybrid
    vector_top_k: int = 10
    hybrid_bm25_weight: float = 0.5
    hybrid_vector_weight: float = 0.5
    vector_model: str = "BAAI/bge-small-en-v1.5"

class RerankerConfig(BaseSettings):
    reranker_type: str = "cross-encoder"  # cross-encoder, llm, none
    cross_encoder_model: str = "models/bge-reranker-vi/final"
    reranker_top_k: int = 5

class DataConfig(BaseSettings):
    data_root: str = "data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

class QueryRewritingConfig(BaseSettings):
    enabled: bool = True
    multi_query_count: int = 3
    hyde_enabled: bool = True
    code_switching_detection: bool = True

class EvaluationConfig(BaseSettings):
    ragas_sample_size: int = 100
    context_max_length: int = 2000

class Config(BaseSettings):
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    data: DataConfig = DataConfig()
    query_rewriting: QueryRewritingConfig = QueryRewritingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    class Config:
        env_file = ".env"

config = Config()