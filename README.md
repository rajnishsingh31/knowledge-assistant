# Knowledge Assistant – Building a Production RAG System from First Principles

A production-oriented Retrieval-Augmented Generation (RAG) application built from first principles.

The objective of this project is to understand and implement the engineering behind modern AI applications instead of relying solely on high-level frameworks. Every component—from document ingestion to grounded answer generation—is implemented incrementally using clean architecture and production software engineering practices.

---

## Why this project?

Most RAG tutorials stop at:

```
Document → Embedding → LLM
```

This project focuses on building the underlying platform:

* Modular architecture
* Provider abstraction
* Explainable retrieval
* Grounded answer generation
* Clean separation of domain and infrastructure
* Local-first development

The long-term goal is to evolve this into a production-ready AI platform supporting hybrid retrieval, reranking, evaluation, memory, and agentic workflows.

---

# Current Features

## Document Ingestion

* Load Markdown (`.md`) and text (`.txt`) documents
* Chunk documents while preserving source location
* Track document IDs and chunk IDs

## Embeddings

* Local embeddings using Sentence Transformers
* Strategy Pattern for embedding providers
* Provider-independent architecture

## Vector Database

* LanceDB
* Apache Arrow storage
* Semantic vector search

## Retrieval

* Semantic similarity search
* Traceable document chunks
* Source line numbers
* Configurable Top-K retrieval

## Grounded Answer Generation

* Local LLM using Ollama
* Prompt construction from retrieved evidence
* Source citations
* Provider abstraction (`LLMProvider`)

## Explainability

* Explain complete RAG execution
* Retrieved chunks
* Prompt sent to the model
* Generated answer
* Active provider configuration

## CLI

* ingest
* search
* ask
* explain
* stats
* inspect

---

# Architecture

```text
                        Knowledge Assistant

                           CLI Commands
                                 │
      ┌──────────────┬────────────┼──────────────┬──────────────┐
      │              │            │              │              │
   ingest         search         ask         explain        inspect
      │              │            │              │              │
      └──────────────┴────────────┴──────────────┘──────────────┘
                                 │                  
                                 ▼
                          Document Loader
                                 │
                                 ▼
                              Chunker
                                 │
                                 ▼
                       Embedding Provider
                     (Sentence Transformers)
                                 │
                                 ▼
                           LanceDB Store
                                 │
                                 ▼
                             Retriever
                                 │
                                 ▼
                        Retrieved Context
                                 │
                                 ▼
                          Prompt Builder
                                 │
                                 ▼
                               Prompt
                                 │
                      ┌──────────┴──────────┐
                      │                     │
                Ollama Provider      OpenAI Provider
                   (Current)        (*Can be extended)
                      │
                      ▼
              Grounded Answer
```

---

# Technology Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Language        | Python 3.13           |
| Package Manager | uv                    |
| Embeddings      | Sentence Transformers |
| Vector Database | LanceDB               |
| Storage Format  | Apache Arrow          |
| Local LLM       | Ollama                |
| Default Model   | qwen3:1.7b            |
| CLI             | argparse              |

---

# Project Structure

```
knowledge-assistant/

├── documents/
├── src/
│   └── knowledge_assistant/
│       ├── answering.py
│       ├── chunking.py
│       ├── document_loader.py
│       ├── embeddings.py
│       ├── formatters.py
│       ├── llm.py
│       ├── main.py
│       ├── models.py
│       ├── prompt_builder.py
│       ├── retrieval.py
│       └── vector_store.py
│
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

# Prerequisites

| Requirement | Version                 |
| ----------- | ----------------------- |
| Python      | 3.13                    |
| uv          | Latest                  |
| Ollama      | Latest                  |
| OS          | Linux / WSL recommended |

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd knowledge-assistant
```

Install dependencies:

```bash
uv sync
```

---

# Local LLM Setup (Ollama)

The `ask` and `explain` commands use a locally running Large Language Model through Ollama.

## Install Ollama

Official installation:

https://ollama.com/download

Linux / WSL:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If installation reports a missing `zstd` dependency:

```bash
sudo apt update
sudo apt install zstd
```

---

## Start Ollama

```bash
ollama serve
```

Leave this terminal running.

---

## Download the default model

```bash
ollama pull qwen3:1.7b
```

---

## Verify Ollama

```bash
ollama run qwen3:1.7b
```

Example:

```
What is Retrieval-Augmented Generation?
```

Exit:

```
/bye
```

---

# Build the Vector Index

Index the default documents:

```bash
uv run knowledge-assistant ingest
```

Index a single document:

```bash
uv run knowledge-assistant ingest documents/python-basics.md
```

Index a different folder:

```bash
uv run knowledge-assistant ingest ./my-notes
```

---

# CLI Commands

## Semantic Search

```bash
uv run knowledge-assistant search "What is BM25?"
```

Specify the number of retrieved chunks:

```bash
uv run knowledge-assistant search "What is BM25?" --limit 5
```

---

## Grounded Question Answering

```bash
uv run knowledge-assistant ask "What is BM25 and when is it useful?"
```

---

## Explain the Complete RAG Pipeline

```bash
uv run knowledge-assistant explain "What is BM25?"
```

Shows:

* retrieved chunks
* retrieval configuration
* prompt sent to the model
* generated answer
* provider information

---

## View Index Statistics

```bash
uv run knowledge-assistant stats
```

Example:

```
Table: knowledge_chunks_minilm_v1
Chunks: 52
Documents: 8
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Dimensions: (384,)
```

---

## Inspect Indexed Chunks

```bash
uv run knowledge-assistant inspect
```

Limit output:

```bash
uv run knowledge-assistant inspect --limit 3
```

---

# Configuration

Current defaults:

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

Future versions will support OpenAI-compatible providers without requiring architectural changes.

---

# Example

```text
$ uv run knowledge-assistant ask \
"What is BM25?"

BM25 is a lexical retrieval algorithm that performs well for
exact identifiers, API names and error codes.

Sources

1.
bm25.md
Lines 14-22

Generated by:
ollama / qwen3:1.7b
```

---

# Engineering Decisions

This project intentionally emphasizes software engineering over framework usage.

Key design principles include:

* Clean Architecture
* Strategy Pattern
* Immutable domain models
* Separation of concerns
* Provider abstraction
* Local-first development
* Explainable AI
* Testability

---

# Roadmap

## Version 0.1 ✅

* Document ingestion
* Chunking
* Local embeddings
* LanceDB integration
* Semantic search
* Grounded answer generation
* Ollama provider
* Explainable retrieval
* CLI

## Planned

* OpenAI provider
* Hybrid Search (BM25 + Vector)
* Reranking
* Retrieval evaluation
* Metadata filtering
* Conversation memory
* REST API
* Multi-document reasoning
* Agentic workflows

---

# Contributing

Issues, suggestions, and pull requests are welcome.

The project is intentionally being developed incrementally to demonstrate production AI engineering practices.

---

# License

MIT License
