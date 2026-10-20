# Advanced RAG Codes

[![guardrails-tests](https://github.com/alalekozai/Advanced-RAG-Codes/actions/workflows/guardrails-tests.yml/badge.svg)](https://github.com/alalekozai/Advanced-RAG-Codes/actions/workflows/guardrails-tests.yml)

A step-by-step collection of Retrieval-Augmented Generation (RAG) code, built with
LangChain and LangGraph. It starts with loading documents and ends with
evaluating and securing a full RAG pipeline. Each module is a self-contained
project with its own dependencies, notebooks, and sample data.

```
load → split → embed → store → retrieve → generate → evaluate → guard
 01     02      03      04       05–11       06–09        10         12
```

## Modules

All modules live under [`Advanced_Rag_Codes_Abdul/`](Advanced_Rag_Codes_Abdul).

| # | Module | What it covers |
|---|---|---|
| 01 | [Document loaders](Advanced_Rag_Codes_Abdul/01_document_loaders) | Loading text, PDF, CSV, JSON, and web pages into LangChain `Document`s |
| 02 | [Text splitters](Advanced_Rag_Codes_Abdul/02_text_splitters) | Character, recursive, document-aware, semantic, and LLM-based chunking |
| 03 | [Embeddings](Advanced_Rag_Codes_Abdul/03_embeddings) | OpenAI embeddings and local embeddings with Ollama |
| 04 | [Vector stores](Advanced_Rag_Codes_Abdul/04_vector_stores) | ChromaDB CRUD, a PDF-to-Chroma pipeline, and reloading a saved DB |
| 05 | [Retrievers](Advanced_Rag_Codes_Abdul/05_retrievers) | Similarity search, score threshold, MMR, BM25, hybrid search, custom ensemble |
| 06 | [RAG Fusion](Advanced_Rag_Codes_Abdul/06_rag_fusion) | LLM-generated sub-queries merged with reciprocal rank fusion |
| 07 | [HyDE](Advanced_Rag_Codes_Abdul/07_rag_hyde) | Hypothetical Document Embeddings: retrieve with a generated answer instead of the raw question |
| 08 | [Agentic RAG](Advanced_Rag_Codes_Abdul/08_agentic_rag) | LangGraph agents: conditional retrieval, tool use with Tavily web search, retrieval grading and query rewriting, query decomposition |
| 09 | [Graph RAG](Advanced_Rag_Codes_Abdul/09_graph_rag) | Building a knowledge graph in Neo4j with an LLM and querying it |
| 10 | [RAG evaluation](Advanced_Rag_Codes_Abdul/10_rag_evaluation) | Scoring a pipeline with Ragas: faithfulness, answer relevancy, context precision and recall, noise sensitivity |
| 11 | [Advanced retrievers](Advanced_Rag_Codes_Abdul/11_advanced_retrievers) | Multi-query, contextual compression, parent-document, and self-query retrievers, each as a built-in and a custom version |
| 12 | [RAG guardrails](Advanced_Rag_Codes_Abdul/12_rag_guardrails) | PII redaction, prompt-injection detection, system-prompt leak detection, and audit logging around the module 10 pipeline |

## Tech stack

- **Frameworks:** LangChain, LangGraph
- **LLMs and embeddings:** OpenAI (`gpt-5-mini`, `text-embedding-3-small`), Ollama for local embeddings
- **Vector store:** ChromaDB
- **Graph database:** Neo4j (AuraDB Free)
- **Web search:** Tavily
- **Evaluation:** Ragas
- **Tooling:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Jupyter

## Getting started

Every module is its own uv project, so set up the one you want to run:

```bash
git clone https://github.com/alalekozai/Advanced-RAG-Codes.git
cd Advanced-RAG-Codes/Advanced_Rag_Codes_Abdul/05_retrievers

uv sync                 # or: pip install -r requirements.txt
```

Then open the notebooks in Jupyter or VS Code and select the module's `.venv` as the kernel.

Module 01 pins Python 3.14 in `.python-version`; the others use 3.12.

### Environment variables

Create a `.env` file inside the module folder. Only add the keys that module needs:

```bash
OPENAI_API_KEY=sk-...        # most modules
TAVILY_API_KEY=tvly-...      # 08_agentic_rag
NEO4J_URI=neo4j+s://...      # 09_graph_rag
NEO4J_USERNAME=neo4j         # 09_graph_rag
NEO4J_PASSWORD=...           # 09_graph_rag
```

For Neo4j setup, see [`09_graph_rag/steps.md`](Advanced_Rag_Codes_Abdul/09_graph_rag/steps.md).
For local embeddings in module 03, install [Ollama](https://ollama.com) and pull an embedding model.

## Tests

The guardrails module has a test suite that runs in CI on every push to it. It
uses a fake retriever and a fake LLM, so it needs no API key:

```bash
cd Advanced_Rag_Codes_Abdul/12_rag_guardrails
uv sync
uv run pytest -v
```

## Author

Waris Alekozai ([@alalekozai](https://github.com/alalekozai))
