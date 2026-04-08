# Vietnamese RAG with Query Rewriting and RankRAG

An advanced Retrieval-Augmented Generation (RAG) system for Vietnamese language, featuring query rewriting techniques and ranking-based document reranking.

## 🎯 Features

- **Query Rewriting**
  - Multi-query expansion
  - HyDE (Hypothetical Document Embeddings)
  - Code-switching detection and handling (VN-EN)

- **Retrieval**
  - Hybrid retrieval (BM25 + Dense vector search)
  - Support for multiple datasets

- **Reranking (RankRAG)**
  - Cross-encoder based reranking
  - LLM-based reranking
  - Hybrid reranking approach

- **Generation**
  - Local LLM support (Ollama with Phi-3, Mistral)
  - Groq API (free, fast)
  - HuggingFace transformers

- **Evaluation**
  - EM, F1 scores
  - Faithfulness and relevance metrics
  - Precision@K and Recall@K

## 📊 Supported Datasets

- **vhealthqa**: Vietnamese health QA dataset
- **uit_viquad2**: UIT Vietnamese Question Answering Dataset
- **vietnamese_rag**: General Vietnamese RAG dataset
- **vimpa**: Vietnamese Multi-domain QA dataset

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repo_url>
cd reranking_rag_and_qw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup LLM

**Option 1: Using Ollama (Recommended for local)**

```bash
# Download Ollama from https://ollama.ai
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull phi      # ~2.5GB
# or
ollama pull mistral  # ~3.8GB
```

**Option 2: Using Groq API (Fast, Free)**

```bash
# Get free API key from https://console.groq.com
# Set environment variable
export GROQ_API_KEY="your_api_key"
# Then update .env file or config
```

**Option 3: Using HuggingFace Models**

Already included in requirements, will auto-download models.

### 3. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env to configure:
# - LLM provider (ollama, groq, huggingface)
# - Model names
# - Retriever and reranker settings
```

### 4. Prepare Data

```bash
# Data should be in format:
# data/{dataset_name}/processed/{split}.jsonl
# Where split is one of: train, validation, test

# Example structure:
# data/vhealthqa/processed/train.jsonl
# data/vhealthqa/processed/validation.jsonl
# data/vhealthqa/processed/test.jsonl
```

## 📖 Usage

### Interactive Mode

```bash
python main.py --interactive
```

Then enter queries interactively.

### Evaluation on Dataset

```bash
# Evaluate on single dataset
python scripts/run_rag_pipeline.py --dataset vhealthqa --split test

# Evaluate on all datasets
python scripts/run_rag_pipeline.py --dataset all --split test

# Specify maximum samples
python scripts/run_rag_pipeline.py --dataset vhealthqa --max_samples 100

# Change LLM provider
python scripts/run_rag_pipeline.py --llm_provider groq --llm_model llama2-70b-4096
```

### Single Query

```python
from main import run_rag_pipeline

answer, metrics = run_rag_pipeline(
    user_query="Machine learning là gì?",
    ground_truth="Machine learning là một lĩnh vực của AI..."
)

print(f"Answer: {answer}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## 🏗️ Architecture

```
Query Input
    ↓
Query Rewriting (Multi-query, HyDE, Code-switching)
    ↓
Retrieval (BM25 + Vector Search)
    ↓
Reranking (Cross-encoder / LLM)
    ↓
Context Selection
    ↓
Generation (LLM)
    ↓
Evaluation (Metrics)
    ↓
Answer Output
```

## 📁 Project Structure

```
reranking_rag_and_qw/
├── src/
│   ├── config/
│   │   └── config.py           # Configuration management
│   ├── data/
│   │   └── data_loader.py      # Data loading and parsing
│   ├── generation/
│   │   └── llm_generator.py    # LLM generation (Ollama, Groq, HF)
│   ├── preprocessing/
│   │   └── code_switching_processor.py  # VN-EN code-switching handling
│   ├── retrieval/
│   │   └── hybrid_retriever.py # BM25 + Vector search
│   ├── reranking/
│   │   └── rank_rag.py         # Cross-encoder / LLM reranking
│   └── rewriting/
│       ├── query_rewriter.py   # Multi-query, HyDE, Code-switching
│       └── prompt_templates.py # Prompt templates
├── evaluation/
│   └── evaluator.py            # Evaluation metrics
├── scripts/
│   └── run_rag_pipeline.py     # Full pipeline script
├── data/
│   └── {dataset_name}/processed/{split}.jsonl
├── results/                    # Evaluation results
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🔧 Configuration Reference

### LLM Configuration

```python
# Ollama (Local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi
OLLAMA_HOST=http://localhost:11434

# Groq (API)
LLM_PROVIDER=groq
GROQ_API_KEY=<your_key>

# HuggingFace
LLM_PROVIDER=huggingface
HF_MODEL_NAME=microsoft/phi-3
HF_QUANTIZATION=4bit
```

### Retriever Configuration

```python
RETRIEVER_TYPE=hybrid  # bm25, vector, or hybrid
VECTOR_TOP_K=10
HYBRID_BM25_WEIGHT=0.5
HYBRID_VECTOR_WEIGHT=0.5
```

### Reranker Configuration

```python
RERANKER_TYPE=cross-encoder  # cross-encoder, llm, or none
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
RERANKER_TOP_K=5
```

## 💾 Hardware Requirements

- **RAM**: 8GB (minimum)
- **VRAM**: 1GB (for quantized models) - CPU inference also supported
- **Disk**: 20GB (for models and data)

### Tested Configurations

| Component | Memory | VRAM | Status |
|-----------|--------|------|--------|
| Phi-3 | 6GB | CPU | ✅ Works |
| Mistral-7B | 8GB | CPU | ✅ Works (slow) |
| Cross-encoder | 1GB | CPU | ✅ Works |
| BM25 Retriever | 2GB | - | ✅ Works |

## 📊 Evaluation Metrics

- **EM (Exact Match)**: Exact string match
- **F1 Score**: Token overlap based
- **Similarity**: String similarity ratio
- **Faithfulness**: How much answer is supported by context
- **Relevance**: How relevant context is to query
- **Precision@K / Recall@K**: Retrieval metrics

## 🔍 Example Output

```
--- Processing: Machine learning là gì? ---

📝 Rewritten queries (4):
   1. Machine learning là gì?
   2. Machine learning định nghĩa
   3. Học máy là gì
   4. Khái niệm machine learning

📚 Retrieved 10 documents
   1. Machine learning là một lĩnh vực... (score: 0.876)
   2. Học máy là quá trình mà các thuật toán... (score: 0.823)
   3. AI và machine learning... (score: 0.756)

🔄 Reranked documents (top 5):
   1. Machine learning là một lĩnh vực... (score: 0.912)
   2. Học máy là quá trình mà các thuật toán... (score: 0.889)
   3. AI và machine learning... (score: 0.834)

💬 Answer: Machine learning là một lĩnh vực của trí tuệ nhân tạo...

📊 Evaluation metrics:
   EM Score: 0.0000
   F1 Score: 0.7234
   Similarity: 0.8123
   Faithfulness: 0.9456
   Relevance: 0.8934
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_retriever.py
```

## 📝 Citation

If you use this project, please cite:

```bibtex
@misc{vietnamese_rag_2024,
  author = {Your Name},
  title = {Vietnamese RAG with Query Rewriting and RankRAG},
  year = {2024},
  url = {https://github.com/yourusername/reranking_rag_and_qw}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📞 Support

- Create an issue on GitHub
- Check documentation in README
- Review example scripts in `scripts/`

## 🙏 Acknowledgments

- Underthesea for Vietnamese NLP
- Ollama for local LLM inference
- HuggingFace for models and transformers
- Groq for fast API inference