# Knowledge Assistant

**A local-first, explainable RAG system built from first principles.**

Knowledge Assistant indexes local documents and supports semantic search, keyword search, hybrid retrieval, and grounded question answering using a locally hosted LLM.

The project intentionally avoids high-level AI orchestration frameworks in its early stages. Its purpose is to demonstrate the engineering behind production RAG systems, including provider abstraction, dependency injection, typed configuration, retrieval strategies, citations, and pipeline explainability.


---

## Current Capabilities

### Document ingestion

* Load Markdown (`.md`) and plain-text (`.txt`) files
* Ingest one file, a directory, or the configured default directory
* Generate stable document and chunk identifiers
* Preserve source file and line-number metadata
* Create overlapping, line-based chunks

### Embeddings and storage

* Generate embeddings locally using Sentence Transformers
* Use a provider-neutral `EmbeddingProvider` abstraction
* Store chunks, vectors, and citation metadata in LanceDB
* Create a LanceDB full-text-search index during ingestion
* Inspect stored chunks and index statistics

### Retrieval

* Vector similarity search
* BM25 full-text search
* Hybrid retrieval
* Reciprocal Rank Fusion (RRF)
* Configurable retrieval strategy
* Per-command retrieval strategy override
* Configurable candidate and result limits

### Answer generation

* Generate answers locally through Ollama
* Use a provider-neutral `LLMProvider` abstraction
* Construct prompts from retrieved evidence
* Require answers to remain grounded in supplied context
* Display supporting source chunks and line numbers
* Keep retrieval commands independent of the LLM

### Explainability

* Display retrieved chunks and ranking metadata
* Display vector distance and BM25 score when available
* Display the complete system and user prompts
* Display the configured embedding model and LLM
* Display the final generated answer

### Retrieval evaluation

* Offline retrieval evaluation framework
* Top-1 and Top-k accuracy metrics
* Compare Vector, BM25, and Hybrid retrieval
* Configurable evaluation dataset
* Per-query evaluation reports
* No LLM required

---

## Architecture

```text
                              CLI
                               │
                               ▼
                 KnowledgeAssistantApplication
                               │
       ┌──────────────┬────────┴────────┬──────────────┐
       │              │                 │              │
       ▼              ▼                 ▼              ▼
   Ingestion      Retrieval        Answering      Evaluation
       │              │                 │              │
       ▼              ▼                 ▼              ▼
Document Loader   Retriever       AnswerService  RetrievalEvaluator
       │              │                 │              │
       ▼              ▼                 ▼              ▼
    Chunker   RetrievalStrategy   PromptBuilder   Evaluation Dataset
       │       ┌─────┼─────┐            │              │
       ▼       │     │     │            ▼              ▼
Embedding   Vector  BM25 Hybrid      Prompt        Metrics
Provider       │      │    │            │
       ▼       └──────┴────┘            ▼
Sentence       Reciprocal Rank      LLMProvider
Transformers      Fusion                │
       │                                ▼
       ▼                           OllamaProvider
  LanceDB
```

### Dependency construction

```text
Settings
   │
   ▼
bootstrap.py
   │
   ├── EmbeddingProvider
   ├── LanceDBVectorStore
   ├── RetrievalStrategy implementations
   ├── Retriever
   ├── LLMProvider
   ├── AnswerService
   └── KnowledgeAssistantApplication
```

`bootstrap.py` is the composition root. Application services receive dependencies through their constructors rather than constructing infrastructure implementations internally.

---

## Technology Stack

| Area                    | Technology             |
| ----------------------- | ---------------------- |
| Language                | Python 3.13            |
| Package management      | uv                     |
| Configuration           | Pydantic Settings      |
| Local embeddings        | Sentence Transformers  |
| Default embedding model | `all-MiniLM-L6-v2`     |
| Vector database         | LanceDB                |
| Keyword retrieval       | LanceDB BM25/FTS       |
| Rank fusion             | Reciprocal Rank Fusion |
| Local LLM runtime       | Ollama                 |
| Default LLM             | `qwen3:1.7b`           |
| CLI                     | argparse               |
| Columnar data           | Apache Arrow           |

---

## Project Structure

```text
knowledge-assistant/
├── documents/
│   └── *.md
├── src/
│   └── knowledge_assistant/
│       ├── answering.py
│       ├── application.py
│       ├── bootstrap.py
│       ├── chunking.py
│       ├── config.py
│       ├── document_loader.py
│       ├── embeddings.py
│       │── evaluation.py 
│       ├── formatters.py
│       ├── llm.py
│       ├── main.py
│       ├── models.py
│       ├── prompt_builder.py
│       ├── retrieval.py
│       └── vector_store.py
├── tests/
├── evaluations/
│   └── retrieval.json
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

The generated LanceDB index is stored under `data/` and is not committed to Git.

---

## Prerequisites

| Requirement      | Recommended version    |
| ---------------- | ---------------------- |
| Python           | 3.13                   |
| uv               | Current stable version |
| Ollama           | Current stable version |
| Operating system | Linux or WSL           |

The retrieval commands work without Ollama. Ollama is required only for `ask` and `explain`.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd knowledge-assistant
```

Create the project environment and install dependencies:

```bash
uv sync
```

Verify Python:

```bash
uv run python --version
```

Expected:

```text
Python 3.13.x
```

---

## Local LLM Setup

The `ask` and `explain` commands use a locally running Ollama model.

### Install Ollama on Linux or WSL

If required, install `zstd` first:

```bash
sudo apt update
sudo apt install zstd
```

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
```

### Start Ollama

```bash
ollama serve
```

If port `11434` is already in use, an Ollama server may already be running.

Check available models:

```bash
ollama list
```

### Download the default model

```bash
ollama pull qwen3:1.7b
```

Test it directly:

```bash
ollama run qwen3:1.7b
```

Exit the interactive session with:

```text
/bye
```

### Ollama process commands

Show currently loaded models:

```bash
ollama ps
```

Unload the model while keeping the server available:

```bash
ollama stop qwen3:1.7b
```

When Ollama was started manually with `ollama serve`, stop the server using `Ctrl+C` in that terminal.

---

## Configuration

The application uses typed settings with sensible defaults. Configuration can be overridden through a local `.env` file or environment variables.

Copy the example file:

```bash
cp .env.example .env
```

Example configuration:

```dotenv
KNOWLEDGE_ASSISTANT_DOCUMENTS__PATH=documents
KNOWLEDGE_ASSISTANT_DOCUMENTS__MAX_CHUNK_LINES=8
KNOWLEDGE_ASSISTANT_DOCUMENTS__OVERLAP_LINES=2

KNOWLEDGE_ASSISTANT_EMBEDDINGS__PROVIDER=sentence-transformers
KNOWLEDGE_ASSISTANT_EMBEDDINGS__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

KNOWLEDGE_ASSISTANT_VECTOR_STORE__PROVIDER=lancedb
KNOWLEDGE_ASSISTANT_VECTOR_STORE__DATABASE_PATH=data/lancedb
KNOWLEDGE_ASSISTANT_VECTOR_STORE__TABLE_NAME=knowledge_chunks_minilm_v1

KNOWLEDGE_ASSISTANT_RETRIEVAL__STRATEGY=hybrid
KNOWLEDGE_ASSISTANT_RETRIEVAL__DEFAULT_LIMIT=3
KNOWLEDGE_ASSISTANT_RETRIEVAL__CANDIDATE_LIMIT=10
KNOWLEDGE_ASSISTANT_RETRIEVAL__RRF_K=60

KNOWLEDGE_ASSISTANT_LLM__PROVIDER=ollama
KNOWLEDGE_ASSISTANT_LLM__MODEL_NAME=qwen3:1.7b
KNOWLEDGE_ASSISTANT_LLM__OLLAMA_HOST=http://localhost:11434
KNOWLEDGE_ASSISTANT_LLM__TEMPERATURE=0
```

Do not commit `.env`. Commit only `.env.example`.

---

## CLI Help

Display application-level help:

```bash
uv run knowledge-assistant --help
```

Expected command groups:

```text
ingest
search
ask
explain
stats
inspect
```

Display help for a specific command:

```bash
uv run knowledge-assistant ingest --help
```

```bash
uv run knowledge-assistant search --help
```

```bash
uv run knowledge-assistant ask --help
```

```bash
uv run knowledge-assistant explain --help
```

```bash
uv run knowledge-assistant inspect --help
```

---

## CLI Commands

### Ingest documents

Index the configured default documents directory:

```bash
uv run knowledge-assistant ingest
```

Index one file:

```bash
uv run knowledge-assistant ingest documents/python-basics.md
```

Index another directory:

```bash
uv run knowledge-assistant ingest ./my-notes
```

Ingestion currently rebuilds the configured vector table from the supplied source. Incremental upsert is planned.

---

### Search documents

Use the configured retrieval strategy:

```bash
uv run knowledge-assistant search "What is BM25?"
```

Limit the number of results:

```bash
uv run knowledge-assistant search \
"What is BM25?" \
--limit 5
```

Use vector retrieval:

```bash
uv run knowledge-assistant search \
"How does the system find text with similar meaning?" \
--strategy vector
```

Use BM25 retrieval:

```bash
uv run knowledge-assistant search \
"CreateSubmissionAsync" \
--strategy bm25
```

Use hybrid retrieval:

```bash
uv run knowledge-assistant search \
"Why is BM25 useful for error codes?" \
--strategy hybrid
```

Supported strategies:

```text
vector
bm25
hybrid
```

When `--strategy` is omitted, the application uses the configured strategy.

---

### Ask a grounded question

```bash
uv run knowledge-assistant ask \
"What is BM25 and when is it useful?"
```

Specify the maximum number of source chunks:

```bash
uv run knowledge-assistant ask \
"What is BM25 and when is it useful?" \
--limit 5
```

The output contains:

* a generated answer,
* source chunks,
* file names,
* line ranges,
* retrieval scores,
* the active LLM provider and model.

Only this command requires the Ollama server.

---

### Explain the RAG pipeline

```bash
uv run knowledge-assistant explain \
"What is BM25?"
```

The command displays:

* the question,
* configured models,
* retrieval method,
* retrieved chunks,
* vector distances,
* BM25 scores,
* hybrid RRF scores,
* system prompt,
* user prompt,
* final generated answer.

This command is intended for local debugging. It may expose full source content and prompts, so production systems should add redaction before exposing comparable traces.

---

### Display index statistics

```bash
uv run knowledge-assistant stats
```

Example:

```text
Table: knowledge_chunks_minilm_v1
Chunks: 52
Documents: 8
Models: sentence-transformers/all-MiniLM-L6-v2
Dimensions: (384,)
```

This command does not require Ollama.

---

### Inspect indexed chunks

```bash
uv run knowledge-assistant inspect
```

Limit output:

```bash
uv run knowledge-assistant inspect --limit 3
```

This command displays stored chunk content and citation metadata without running a retrieval query.

---

### Evaluate Retrieval

Evaluate all retrieval strategies:

```bash
uv run knowledge-assistant evaluate
```

Evaluate one strategy:

```bash
uv run knowledge-assistant evaluate --strategy hybrid
```

Show detailed per-query results:

```bash
uv run knowledge-assistant evaluate --details
```

Use a custom evaluation dataset:

```bash
uv run knowledge-assistant evaluate \
    --dataset evaluations/retrieval.json
```

## Retrieval Strategies

### Vector retrieval

The query and chunks are represented as embeddings in the same vector space. LanceDB returns chunks with the smallest vector distance.

Best suited to:

* semantic similarity,
* paraphrased questions,
* concept-based queries,
* wording that differs from source documents.

### BM25 retrieval

BM25 performs lexical full-text retrieval over chunk content.

Best suited to:

* exact identifiers,
* API names,
* error codes,
* model names,
* rare technical terms.

### Hybrid retrieval

Hybrid retrieval executes vector and BM25 retrieval independently and combines their rank positions using Reciprocal Rank Fusion.

```text
Vector ranked list ─┐
                    ├── Reciprocal Rank Fusion ── Final ranking
BM25 ranked list ───┘
```

RRF avoids directly comparing incompatible raw vector-distance and BM25-score scales.

---

## Example Search Output

```text
1. bm25.md
Lines: 9-16
Method: hybrid
Score: 0.032522
Vector distance: 0.8421
BM25 score: 3.1452
------------------------------------------------------------
BM25 combines term-frequency and inverse-document-frequency
signals with document-length normalization.
```

For vector distance, lower values indicate closer matches.

The hybrid score is an RRF ranking score, not a probability or percentage.

---

## Example Grounded Answer

```text
$ uv run knowledge-assistant ask \
"What is BM25 and when is it useful?"

BM25 is a lexical retrieval algorithm that is particularly useful
for exact terms, identifiers, API names, and error codes [Source 1].

Sources:

1. bm25.md
Lines: 9-16
Method: hybrid
...

Generated by: ollama/qwen3:1.7b
```

Generated answers may paraphrase source text. Every material claim should still be supported by the displayed source context.

---

## Engineering Decisions

### Local-first architecture

Document loading, chunking, embeddings, vector storage, BM25 retrieval, and rank fusion run locally.

A hosted API key is not required for the current implementation.

### Provider abstractions

`EmbeddingProvider` and `LLMProvider` separate application logic from concrete AI providers.

Planned providers can be introduced without changing the retrieval or answering services.

### Retrieval Strategy Pattern

`Retriever` depends on a `RetrievalStrategy`, allowing vector, BM25, and hybrid implementations to be selected through configuration or a CLI override.

### Dependency injection

Concrete dependencies are created in `bootstrap.py` and passed through constructors.

No dependency-injection framework is used.

### Typed configuration

Pydantic Settings validates configuration and supports defaults, `.env` files, and environment-variable overrides.

### Domain and storage separation

Domain models such as `Chunk` and `Embedding` remain independent of LanceDB. The vector-store adapter creates a denormalized storage representation optimized for retrieval.

### Explainability

The `explain` command exposes retrieval and generation inputs so incorrect answers can be traced to retrieval quality, prompt construction, or model behavior.

### Measuring Retrieval Quality 

Before introducing new retrieval algorithms such as reranking, the project establishes an offline evaluation framework to measure retrieval quality objectively. This allows improvements to be validated using reproducible metrics rather than subjective observation.

---

## Development Workflow

After modifying the documents or chunking configuration, rebuild the index:

```bash
uv run knowledge-assistant ingest
```

Compare retrieval strategies:

```bash
uv run knowledge-assistant search \
"Why is BM25 useful for error codes?" \
--strategy vector
```

```bash
uv run knowledge-assistant search \
"Why is BM25 useful for error codes?" \
--strategy bm25
```

```bash
uv run knowledge-assistant search \
"Why is BM25 useful for error codes?" \
--strategy hybrid
```

Inspect the complete pipeline:

```bash
uv run knowledge-assistant explain \
"Why is BM25 useful for error codes?"
```

---

## Roadmap

### Completed

* Local document ingestion
* Traceable chunking
* Local embeddings
* LanceDB vector storage
* BM25 full-text search
* Vector retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion
* Retrieval strategy CLI override
* Typed configuration
* Constructor-based dependency injection
* Grounded answer generation
* Ollama integration
* Source citations
* Explainable RAG traces
* CLI operations
* Retrieval evaluation dataset
* Top-1 and Top-k retrieval metrics
* Configurable evaluation dataset
* Strategy comparison (Vector, BM25, Hybrid)

### Next

* Cross-encoder reranking
* Before-and-after evaluation reports
* Unit and integration tests
* Prompt versioning
* Structured logging and latency metrics

### Later

* Incremental document ingestion
* Metadata filtering
* PDF and Word support
* OpenAI provider
* Conversation memory
* REST API
* Web interface
* Multi-document reasoning
* Agentic workflows

---

## Git Hygiene

Commit:

```text
src/
documents/
tests/
pyproject.toml
uv.lock
README.md
.env.example
.python-version
.gitignore
```

Do not commit:

```text
.venv/
.env
data/
__pycache__/
.pytest_cache/
.vscode/
.idea/
```

The vector database is a generated artifact and can be recreated with:

```bash
uv run knowledge-assistant ingest
```

---

## License

MIT
