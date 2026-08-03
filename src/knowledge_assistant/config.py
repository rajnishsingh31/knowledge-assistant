from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocumentSettings(BaseModel):
    path: Path = Path("documents")
    max_chunk_lines: int = Field(default=8, gt=0)
    overlap_lines: int = Field(default=2, ge=0)


class EmbeddingSettings(BaseModel):
    provider: Literal["sentence-transformers"] = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStoreSettings(BaseModel):
    provider: Literal["lancedb"] = "lancedb"
    database_path: Path = Path("data/lancedb")
    table_name: str = "knowledge_chunks_minilm_v1"


class RetrievalSettings(BaseModel):
    strategy: Literal["vector", "bm25", "hybrid"] = "hybrid"
    default_limit: int = Field(default=3, gt=0)
    candidate_limit: int = Field(default=10, gt=0)
    rrf_k: int = Field(default=60, gt=0)


class LLMSettings(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    model_name: str = "qwen3:1.7b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

class EvaluationSettings(BaseModel):
    dataset_path: Path = Path("evaluations/retrieval.json")
    top_k: int = Field(default=3, gt=0)

class RerankingSettings(BaseModel):
    strategy: Literal["identity", "cross-encoder"] = "cross-encoder"
    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    retrieval_limit: int = Field(default=10, gt=0)
    final_limit: int = Field(default=3, gt=0)


class Settings(BaseSettings):
    """Application configuration loaded from defaults and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KNOWLEDGE_ASSISTANT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    documents: DocumentSettings = DocumentSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    reranking: RerankingSettings = RerankingSettings()

@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings."""

    return Settings()