# GraphRAG vs RAG vs LLM — Financial Q&A Benchmark

TigerGraph GraphRAG Inference Hackathon submission.

## What This Does

Compares 3 approaches to answering financial questions from SEC 10-K filings:

- **LLM Only** — direct Groq inference, no retrieval
- **Basic RAG** — ChromaDB vector search, top-5 chunks
- **GraphRAG** — Knowledge graph traversal, precise entity-based retrieval

## Results

| Metric | RAG | GraphRAG |
|--------|-----|----------|
| Avg Tokens | ~1,456 | ~979 |
| Token Reduction | baseline | **31.9%** |
| BERTScore F1 | 0.821 | **0.822** |
| LLM Judge Pass | — | **70%** |

## Dataset

- PatronusAI/FinanceBench + Finance-Alpaca
- 2.1M tokens · 11,646 chunks · SEC 10-K filings

## Stack

- LLM: Groq llama3-70b-8192
- Graph DB: TigerGraph Savanna
- Vector DB: ChromaDB
- Embeddings: sentence-transformers
- Backend: FastAPI

## Setup

```bash
pip install -r requirements.txt
python ingest.py
python backend/evaluate.py
uvicorn backend.main:app --reload
