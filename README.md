# Knowledge Assistant – Building a Production RAG System from First Principles

A production-oriented Retrieval-Augmented Generation (RAG) application built from first principles.

The goal of this project is not only to answer questions over documents, but also to demonstrate the engineering practices behind modern AI applications including document ingestion, semantic search, vector databases, clean architecture, and agentic workflows.

The project is being built incrementally, with each milestone introducing one production concept at a time.

## Learning Goals

This project is intentionally built without relying on high-level AI frameworks in the early stages.

The objective is to understand how production AI systems work internally before introducing orchestration frameworks or managed services.

Every component is implemented incrementally to emphasize software engineering principles such as modularity, clean architecture, immutability, separation of concerns, and testability.

## Current Features

- Load Markdown and text documents
- Split documents into traceable chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in LanceDB
- Perform semantic search over indexed documents
- Command-line interface for indexing and searching
- Source citations with document names and line numbers
    
## Architecture

```text
                 CLI
                  │
        ┌─────────┴─────────┐
        │                   │
     Ingest              Search
        │                   │
        ▼                   ▼
Document Loader        Retriever
        │                   │
        ▼                   ▼
     Chunker         Embedding Provider
        │                   │
        └─────────┬─────────┘
                  ▼
          LanceDB Vector Store
                  │
                  ▼
            Search Results
```

## Technology Stack

- Python 3.13
- uv
- Sentence Transformers
- LanceDB
- Apache Arrow

## Project Structure

```text
src/
    knowledge_assistant/
        chunking.py
        document_loader.py
        embeddings.py
        formatters.py
        main.py
        models.py
        retrieval.py
        vector_store.py
documents/
data/
```

## Getting Started

### Clone the repository
git clone <repository-url>
cd knowledge-assistant

### Install dependencies
uv sync

### CLI Commands

### Build or rebuild the vector index

- Indexes all supported documents in the default documents/ folder.
  - uv run knowledge-assistant ingest
- Index a specific document:
  - uv run knowledge-assistant ingest documents/python-basics.md
- Index a different folder:
  - uv run knowledge-assistant ingest ./my-notes

### Search the knowledge base

- Search using natural language:
  - uv run knowledge-assistant search "What is Retrieval-Augmented Generation?"
- Limit the number of returned chunks:
  - uv run knowledge-assistant search "What is BM25?" --limit 5

### View index statistics

- Display summary information about the vector index.
  - uv run knowledge-assistant stats
    - Example output:
```text
        Table: knowledge_chunks_minilm_v1
        Chunks: 52
        Documents: 8
        Models: sentence-transformers/all-MiniLM-L6-v2
        Dimensions: (384,)
        Inspect indexed chunks
```
### Display indexed chunks without performing a search.
  - uv run knowledge-assistant inspect
  
  Limit the number of displayed chunks:
  - uv run knowledge-assistant inspect --limit 3

## Example
```text
$ uv run knowledge-assistant search "What does BM25 do?"

1. bm25.md
Lines: 15-24
Distance: 0.89

BM25 is a lexical ranking algorithm that performs well for exact
keywords, identifiers, API names and error codes.
```
## Roadmap

### Phase 1 – Retrieval Foundation ✅
- Document Loader
- Chunking
- Embeddings
- LanceDB Integration
- Semantic Search
- CLI

### Phase 2 – Answer Generation
- Prompt Builder
- LLM Integration
- Grounded Responses
- Citations

### Phase 3 – Advanced Retrieval
- Hybrid Search (BM25 + Vector)
- Reranking
- Metadata Filtering
- Retrieval Evaluation

### Phase 4 – Agentic Workflows
- Tool Calling
- Planning
- Memory
- Multi-step Reasoning

## License

MIT