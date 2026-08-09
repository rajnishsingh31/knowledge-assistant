# Knowledge Assistant

**A production-style, local-first knowledge retrieval and agent platform built from first principles.**

Knowledge Assistant ingests local documents, indexes them for lexical and semantic retrieval, generates grounded answers using a locally hosted LLM, and exposes the same application capabilities through **CLI, REST API, interactive chat, and MCP**.

The project intentionally avoids high-level AI orchestration frameworks. Core RAG and agent components—including retrieval strategies, rank fusion, reranking, planning, tool execution, grounding validation, conversation context, evaluation, and tracing—are implemented explicitly to demonstrate how production AI systems work beneath framework abstractions.

---

## Highlights

### Retrieval and RAG

* Multi-format document ingestion: Markdown, text, PDF, Word, and Excel
* Incremental ingestion with document change detection and embedding reuse
* Structure-aware and line-based chunking
* Local Sentence Transformer embeddings
* LanceDB vector and full-text indexes
* Vector, BM25, and hybrid retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Metadata filtering
* Source and line-level citations
* Grounded local answer generation through Ollama

### Agent Runtime

* Planner / Executor architecture
* Typed tool specifications
* Multi-step tool execution
* Runtime policy enforcement and safeguards
* Independent answer synthesis
* NLI-based grounding validation
* Bounded multi-turn conversation context
* Interactive chat
* Full execution traces

### Quality and Evaluation

* Offline retrieval evaluation
* Vector / BM25 / Hybrid comparison
* Top-1 and Top-k retrieval metrics
* Agent tool-sequence validation
* Expected-document validation
* Grounding evaluation
* Iteration and latency measurement
* Structured observability and pipeline timings
* Automated pytest suite

### Interfaces

* Command-line interface
* Interactive chat
* FastAPI REST API
* OpenAPI / Swagger
* Model Context Protocol (MCP) server

---

## Architecture

```text
                         Users / AI Clients
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
         CLI                REST API              MCP Server
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                    KnowledgeAssistantApplication
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             │                                     │
      Retrieval / RAG                        Agent Runtime
             │                                     │
     ┌───────┼────────┐                    ┌───────┴────────┐
     │       │        │                    │                │
   Vector   BM25    Hybrid               Planner          Policy
                      │                    │                │
                      ▼                    └───────┬────────┘
                     RRF                           │
                      │                            ▼
                      ▼                           Tools
                 Reranker                          │
                      │                            ▼
                      │                        Evidence
                      │                            │
                      └──────────────┐             ▼
                                     │        Synthesizer
                                     │             │
                                     │             ▼
                                     │     Grounding Validator
                                     │             │
                                     └─────────────┴─────► Response
```

### Composition Root

Concrete infrastructure dependencies are constructed centrally in `bootstrap.py`.

```text
Settings
   │
   ▼
bootstrap.py
   │
   ├── DocumentService
   ├── ChunkingStrategy
   ├── EmbeddingProvider
   ├── LanceDBVectorStore
   ├── RetrievalStrategy implementations
   ├── Retriever
   ├── Reranker
   ├── LLMProvider
   ├── AnswerService
   ├── Agent tools
   ├── Agent planner
   ├── Synthesizer
   └── Grounding validator
```

Application and runtime components receive dependencies through constructors rather than creating infrastructure implementations internally.

---

## Engineering Principles

The project is designed around several production-oriented principles:

* **Local-first execution** — documents, embeddings, retrieval, reranking, and LLM inference remain local.
* **Separation of concerns** — ingestion, retrieval, answering, agent orchestration, evaluation, transport, and infrastructure remain independently testable.
* **Provider abstraction** — embedding and LLM implementations are hidden behind typed interfaces.
* **Explicit orchestration** — agent planning and execution are implemented directly instead of delegated to an orchestration framework.
* **Evidence-first generation** — retrieved observations remain authoritative for factual answers.
* **Independent validation** — answer generation and grounding verification use separate components.
* **Observability** — important retrieval, generation, execution, and latency signals are inspectable.
* **Evaluation before optimization** — retrieval and agent behavior are measured through reproducible evaluation suites.

---

## Why No LangChain or LangGraph?

This project intentionally implements the core AI pipeline without high-level orchestration frameworks.

The goal is to understand and demonstrate the underlying engineering:

```text
Retrieval
→ Reranking
→ Planning
→ Tool Execution
→ Evidence Selection
→ Synthesis
→ Grounding
→ Evaluation
```

This makes framework abstractions easier to reason about and avoids hiding important execution behavior behind libraries.

---

# Core Capabilities

## Document Ingestion

Supported formats:

* Markdown (`.md`)
* Plain text (`.txt`)
* PDF (`.pdf`)
* Microsoft Word (`.docx`)
* Microsoft Excel (`.xlsx`)

The ingestion pipeline supports:

* Single-file and directory ingestion
* Stable document identifiers
* Stable chunk identifiers
* Content hashing
* Source-path metadata
* Line-range metadata
* Incremental synchronization
* Deleted-document cleanup
* Unchanged-document skipping
* Chunk-level embedding reuse
* Ingestion timing metrics

### Chunking

Two strategies are available:

**Line-based chunking**

* Fixed maximum line count
* Configurable overlap

**Structure-aware chunking**

* Markdown heading boundaries
* PDF page boundaries
* Excel worksheet boundaries
* Word table markers
* Fixed-line fallback for oversized sections

---

## Embeddings and Storage

Embeddings are generated locally using Sentence Transformers.

The current architecture uses:

```text
EmbeddingProvider
       │
       ▼
SentenceTransformerEmbeddingProvider
```

Indexed content is persisted in LanceDB together with:

* Chunk content
* Vector embeddings
* Document identifiers
* Source paths
* Line ranges
* Content hashes
* Embedding metadata

A full-text-search index is also created for lexical retrieval.

---

## Retrieval

Three retrieval strategies are supported.

### Vector Retrieval

Semantic retrieval using embedding similarity.

Best suited to:

* Semantic similarity
* Paraphrased queries
* Conceptual questions
* Wording that differs from the source

### BM25 Retrieval

Lexical retrieval using LanceDB full-text search.

Best suited to:

* Exact identifiers
* API names
* Error codes
* Model names
* Rare technical terminology

### Hybrid Retrieval

Runs vector and BM25 retrieval independently and combines their ranked results using **Reciprocal Rank Fusion**.

```text
Vector ranking ─┐
                ├── Reciprocal Rank Fusion ──► Final ranking
BM25 ranking ───┘
```

RRF combines rank positions rather than directly comparing incompatible raw vector and BM25 score scales.

---

## Cross-Encoder Reranking

Candidate results can be reranked before answer generation.

The default implementation uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The reranking layer is abstracted behind a `Reranker` interface and includes:

* Cross-encoder reranking
* Identity/no-op reranking
* Configurable candidate count
* Configurable final result count
* Optional reranking during retrieval evaluation

---

## Metadata Filtering

Retrieval can be restricted using:

* Source filename
* File extension

Filtering is supported by:

* `search`
* `ask`
* `explain`
* REST retrieval endpoints

---

## Grounded Answer Generation

Grounded answers are generated locally through Ollama.

The answering pipeline is:

```text
Query
  │
  ▼
Retrieve Candidates
  │
  ▼
Rerank Evidence
  │
  ▼
Build Prompt
  │
  ▼
Local LLM
  │
  ▼
Answer + Sources
```

Generated answers retain their supporting `SearchResult` objects so source documents and line ranges remain traceable.

---

# Agent Runtime

Knowledge Assistant also includes a lightweight multi-step agent runtime built directly on top of the RAG platform.

```text
User Request
     │
     ▼
Planner
     │
     ▼
Policy Enforcement
     │
     ▼
Tool Selection
     │
     ▼
Tool Execution
     │
     ▼
Observation
     │
     ▼
Planner
     │
     ▼
Final Completion Decision
     │
     ▼
Synthesizer
     │
     ▼
Grounding Validator
     │
     ▼
Final Response
```

The planner does not directly own final factual answer generation once evidence has been collected. The synthesizer produces the response from selected observations.

### Agent Tools

The runtime currently includes tools for:

* Searching documents
* Retrieving focused answer evidence
* Inspecting index statistics
* Inspecting raw indexed data

### Runtime Policies

Runtime policies can override planner behavior when necessary to ensure evidence is collected before a factual response is finalized.

Additional safeguards include:

* Maximum iteration limits
* Repeated tool-call detection
* Tool registration validation
* Typed tool arguments
* Execution tracing

---

## Grounding Validation

Final synthesized answers are independently checked using an NLI model:

```text
cross-encoder/nli-deberta-v3-small
```

The validator:

1. Splits the generated response into factual candidate claims.
2. Normalizes formatting and citations.
3. Compares each claim against retrieved evidence.
4. Computes entailment scores.
5. Marks unsupported claims in the execution trace.

Example trace:

```text
GROUNDING VALIDATION
------------------------------------------------------------
Status: FAILED
Checked claims: 5
Supported: 3
Unsupported: 2

Unsupported claims:
- "This approach reduces unauthorized access."
  Reason: No supplied evidence explicitly entails this claim.
```

Grounding validation is intentionally independent from answer generation.

---

## Conversation Context

Interactive agent sessions retain a bounded recent conversation history.

```text
> What is BM25?

> How is it different from vector search?

> Give me an example.
```

Previous conversation turns are supplied to the planner for reference resolution, while tool observations remain authoritative for grounded factual answers.

Current scope:

* Session-local memory
* Bounded recent messages
* Planner context

Not currently implemented:

* Persistent cross-session memory
* Semantic long-term memory
* Episodic memory
* Advanced standalone-query rewriting

---

# Evaluation

## Retrieval Evaluation

The retrieval evaluation framework measures deterministic retrieval quality without requiring an LLM.

Metrics include:

* Top-1 accuracy
* Top-k accuracy
* Expected document retrieval

Supported comparisons:

* Vector
* BM25
* Hybrid
* Hybrid + reranker

Run all strategies:

```bash
uv run knowledge-assistant evaluate
```

Evaluate hybrid retrieval with reranking:

```bash
uv run knowledge-assistant evaluate \
    --strategy hybrid \
    --rerank \
    --details
```

---

## Agent Evaluation

The end-to-end agent evaluation framework measures behavioral regressions across the complete agent pipeline.

Evaluation cases can validate:

* Expected tool sequence
* Expected supporting documents
* Stop reason
* Grounding result
* Iteration count
* End-to-end latency

Run the default suite:

```bash
uv run knowledge-assistant evaluate-agent
```

Include case-level details:

```bash
uv run knowledge-assistant evaluate-agent --details
```

Tool, document, stop-reason, and grounding accuracy are reported separately so failures can be localized to a specific stage of execution.

---

# Interfaces

## Command-Line Interface

Display help:

```bash
uv run knowledge-assistant --help
```

Primary commands:

```text
ingest
rebuild
search
ask
explain
stats
inspect
evaluate
agent
chat
evaluate-agent
```

### Ingest

```bash
uv run knowledge-assistant ingest
```

Single file:

```bash
uv run knowledge-assistant ingest documents/python-basics.md
```

Different directory:

```bash
uv run knowledge-assistant ingest ./my-notes
```

### Rebuild

```bash
uv run knowledge-assistant rebuild
```

### Search

```bash
uv run knowledge-assistant search "What is BM25?"
```

Choose retrieval strategy:

```bash
uv run knowledge-assistant search \
    "CreateSubmissionAsync" \
    --strategy bm25
```

Filter by source:

```bash
uv run knowledge-assistant search \
    "What is least privilege?" \
    --file cloud-security.docx
```

### Ask

```bash
uv run knowledge-assistant ask \
    "What is BM25 and when is it useful?"
```

Filter by file type:

```bash
uv run knowledge-assistant ask \
    "What happens during containment?" \
    --type pdf
```

### Explain

Inspect the complete RAG pipeline:

```bash
uv run knowledge-assistant explain \
    "What is BM25?"
```

The output includes:

* Retrieved chunks
* Ranking metadata
* Vector distances
* BM25 scores
* Hybrid scores
* Reranking
* Prompt construction
* Configured models
* Final answer
* Startup timings
* Pipeline timings

### Agent

```bash
uv run knowledge-assistant agent \
    "Find evidence about least privilege and explain it."
```

Show execution trace:

```bash
uv run knowledge-assistant agent \
    "Find evidence about least privilege and explain it." \
    --trace
```

### Interactive Chat

```bash
uv run knowledge-assistant chat
```

Exit using:

```text
exit
```

or:

```text
quit
```

### Index Statistics

```bash
uv run knowledge-assistant stats
```

### Inspect Index

```bash
uv run knowledge-assistant inspect --limit 5
```

---

# REST API

The FastAPI interface shares the same application layer as the CLI.

Features include:

* Pydantic request/response validation
* Lifecycle-based dependency initialization
* OpenAPI schema generation
* Swagger UI
* Retrieval filters
* Incremental ingestion
* Local grounded answer generation

## Endpoints

| Method | Endpoint   | Purpose                        |
| ------ | ---------- | ------------------------------ |
| `GET`  | `/health`  | Service health                 |
| `GET`  | `/stats`   | Index statistics               |
| `POST` | `/search`  | Search document chunks         |
| `POST` | `/ask`     | Generate a grounded answer     |
| `POST` | `/ingest`  | Incrementally ingest documents |
| `POST` | `/rebuild` | Rebuild the index              |

Start the API:

```bash
uv run knowledge-assistant-api
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

OpenAPI:

```text
http://127.0.0.1:8000/openapi.json
```

Example search:

```bash
curl -X POST \
    http://127.0.0.1:8000/search \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What is least privilege?",
      "limit": 3,
      "strategy": "hybrid",
      "filters": {
        "source_names": ["cloud-security.docx"]
      }
    }'
```

---

# MCP Server

Knowledge Assistant exposes selected application capabilities through the **Model Context Protocol (MCP)**.

The MCP layer is intentionally thin:

```text
MCP Client
    │
    ▼
Knowledge Assistant MCP Server
    │
    ▼
KnowledgeAssistantApplication
    │
    ├── Search
    ├── Answer
    ├── Statistics
    └── Index Inspection
```

No retrieval or business logic is duplicated inside the MCP adapter.

## MCP Tools

| Tool               | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `search_documents` | Retrieve relevant indexed passages                      |
| `answer_question`  | Generate a grounded answer using indexed documents      |
| `get_index_stats`  | Return document, chunk, embedding, and index statistics |
| `inspect_index`    | Inspect raw records stored in the index                 |

Start the MCP server using stdio transport:

```bash
uv run knowledge-assistant-mcp
```

For development with MCP Inspector:

```bash
uv run mcp dev src/knowledge_assistant/mcp/server.py
```

The Inspector requires Node.js / npm / `npx`.

The MCP implementation uses the same application facade as the CLI and REST API, demonstrating transport-independent application logic.

---

# Observability

The project records timing and execution information across key stages.

### Application startup

* Settings loading
* Dependency construction
* Total startup time

### RAG pipeline

* Retrieval latency
* Reranking latency
* Prompt construction latency
* LLM generation latency
* Total pipeline latency

### Agent runtime

* Iteration number
* Planner decisions
* Tool selection
* Tool arguments
* Tool observations
* Completion reason
* Synthesizer identity
* Grounding validation result

Enable diagnostic logs:

```bash
uv run knowledge-assistant --verbose ask "What is BM25?"
```

Verbose traces can expose prompts and source content and should therefore be treated as diagnostic output rather than production-safe telemetry.

---

# Technology Stack

| Area                             | Technology                               |
| -------------------------------- | ---------------------------------------- |
| Language                         | Python 3.13                              |
| Package / environment management | uv                                       |
| Configuration                    | Pydantic Settings                        |
| Embeddings                       | Sentence Transformers                    |
| Default embedding model          | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database                  | LanceDB                                  |
| Lexical retrieval                | LanceDB FTS / BM25                       |
| Hybrid fusion                    | Reciprocal Rank Fusion                   |
| Reranker                         | Cross Encoder                            |
| Default reranker                 | `cross-encoder/ms-marco-MiniLM-L6-v2`    |
| Local LLM runtime                | Ollama                                   |
| Default LLM                      | `qwen3:1.7b`                             |
| Grounding validation             | NLI Cross Encoder                        |
| Default grounding model          | `cross-encoder/nli-deberta-v3-small`     |
| REST API                         | FastAPI                                  |
| MCP                              | MCP Python SDK                           |
| CLI                              | argparse                                 |
| Testing                          | pytest                                   |

---

# Project Structure

```text
knowledge-assistant/
├── documents/
├── evaluations/
│   └── retrieval.json
├── tools/
│   └── generate_sample_documents.py
│
├── src/
│   └── knowledge_assistant/
│       ├── agent/
│       │   ├── evaluation/
│       │   ├── tools/
│       │   ├── citations.py
│       │   ├── evidence.py
│       │   ├── formatting.py
│       │   ├── guards.py
│       │   ├── models.py
│       │   ├── policy.py
│       │   ├── registry.py
│       │   └── runtime.py
│       │
│       ├── api/
│       │   ├── app.py
│       │   ├── dependencies.py
│       │   ├── mappers.py
│       │   ├── schemas.py
│       │   └── server.py
│       │
│       ├── cli/
│       │   └── chat.py
│       │
│       ├── conversation/
│       │   ├── memory.py
│       │   └── models.py
│       │
│       ├── document_loaders/
│       │
│       ├── llm/
│       │   ├── grounding_validator.py
│       │   ├── models.py
│       │   ├── nli_grounding_validator.py
│       │   ├── planner.py
│       │   ├── planner_prompts.py
│       │   ├── prompt_builder.py
│       │   ├── providers.py
│       │   ├── synthesizer.py
│       │   └── synthesis_prompts.py
│       │
│       ├── mcp/
│       │   ├── main.py
│       │   ├── serializers.py
│       │   └── server.py
│       │
│       ├── answering.py
│       ├── application.py
│       ├── bootstrap.py
│       ├── chunking.py
│       ├── config.py
│       ├── document_loader.py
│       ├── embeddings.py
│       ├── evaluation.py
│       ├── formatters.py
│       ├── main.py
│       ├── models.py
│       ├── reranking.py
│       ├── retrieval.py
│       └── vector_store.py
│
├── tests/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

The LanceDB index is generated under `data/` and is not committed to source control.

---

# Prerequisites

| Requirement      | Version        |
| ---------------- | -------------- |
| Python           | 3.13           |
| uv               | Current stable |
| Ollama           | Current stable |
| Operating system | Linux / WSL    |

Retrieval-only commands do not require Ollama.

Ollama is required for functionality that performs local LLM generation, including grounded answering and agent synthesis.

Node.js / npm is required only when using MCP Inspector.

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

Verify Python:

```bash
uv run python --version
```

Expected:

```text
Python 3.13.x
```

---

# Local LLM Setup

Install and start Ollama.

Verify:

```bash
ollama --version
```

Download the configured default model:

```bash
ollama pull qwen3:1.7b
```

Start Ollama:

```bash
ollama serve
```

Check models:

```bash
ollama list
```

---

# Configuration

Configuration is validated using Pydantic Settings and can be overridden using `.env` or environment variables.

Create local configuration:

```bash
cp .env.example .env
```

Example:

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

Never commit `.env`.

---

# Development

Run the complete test suite:

```bash
uv run pytest
```

Run a focused package:

```bash
uv run pytest tests/agent -v
```

Inspect a retrieval pipeline:

```bash
uv run knowledge-assistant explain \
    "Why is BM25 useful for error codes?"
```

Evaluate retrieval:

```bash
uv run knowledge-assistant evaluate \
    --strategy hybrid \
    --rerank
```

Evaluate agent behavior:

```bash
uv run knowledge-assistant evaluate-agent --details
```

Test MCP integration:

```bash
uv run mcp dev src/knowledge_assistant/mcp/server.py
```

---

# Known Limitations

The current implementation is intentionally optimized for learning, local execution, and architectural transparency rather than low-latency production serving.

Current limitations include:

* Local LLM execution can be slow on CPU-constrained environments.
* Grounding validation is conservative and may reject valid paraphrases.
* Conversation history is session-local and bounded.
* Follow-up query resolution is planner-driven and may occasionally require more explicit wording.
* MCP currently exposes core knowledge capabilities rather than the complete agent runtime.
* Authentication, authorization, rate limiting, and distributed deployment are outside the current scope.
* Verbose diagnostic traces are not redacted for production use.

---

# Future Work

Potential extensions include:

* Streaming API and agent responses
* Persistent conversation memory
* Semantic / episodic memory
* Stronger grounding and claim verification models
* Agent evaluation datasets at larger scale
* Authentication and authorization
* Distributed retrieval / storage
* Full agent exposure through MCP
* Multi-agent workflows

---

# Git Hygiene

Commit:

```text
src/
documents/
evaluations/
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

Generated vector indexes can be recreated with:

```bash
uv run knowledge-assistant ingest
```

---

# License

MIT
