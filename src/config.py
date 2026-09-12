"""Configuration settings for TheoremLens Local Math RAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
    # Base directories
    base_dir: Path = Path(__file__).parent.parent.resolve()
    persist_directory: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "chroma_db")
    upload_directory: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "uploads")
    collection_name: str = "math_rag_collection"

    # Hardware & LLM Settings (GPU: RTX 2060 Super 8GB)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:7b"
    llm_num_ctx: int = 8192  # 8192 default, downscale to 6144 or 4096 if VRAM throttles
    llm_temperature: float = 0.0

    # CPU Models Settings
    # Embedding: nomic-embed-text (CPU mode, 768 dim)
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int = 768

    # Reranker: bge-reranker-base (CPU mode)
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_threshold: float = 0.35  # CRAG style guardrail cutoff

    # Chunking Settings (Parent-Child MVP)
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 150
    parent_separators: List[str] = field(
        default_factory=lambda: ["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    child_chunk_size: int = 250
    child_chunk_overlap: int = 40

    # Retrieval Settings
    dense_top_k: int = 10
    sparse_top_k: int = 10
    final_top_parents: int = 2  # 1~2 top parent chunks injected into prompt
    bm25_token_pattern: str = r"\\[a-zA-Z]+|[a-zA-Z0-9]+|[^\s\w]"

    # Gradio Web Server Settings
    server_port: int = 7860
    server_name: str = "127.0.0.1"


config = AppConfig()
config.persist_directory.mkdir(parents=True, exist_ok=True)
config.upload_directory.mkdir(parents=True, exist_ok=True)
