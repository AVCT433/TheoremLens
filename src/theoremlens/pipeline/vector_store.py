"""Chroma Vector Store management with CPU-based nomic-embed-text embeddings."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from theoremlens.config import config
from theoremlens.ingestion import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


def get_default_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Returns official ChromaDB SentenceTransformerEmbeddingFunction configured for CPU nomic-embed-text."""
    logger.info(
        f"Initializing SentenceTransformerEmbeddingFunction ('{config.embedding_model_name}' on {config.embedding_device.upper()})..."
    )
    return SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model_name,
        device=config.embedding_device,
        normalize_embeddings=True,
        trust_remote_code=True,
    )


class ChromaStore:
    """Manages Chroma persistent storage for child chunks and parent metadata."""

    def __init__(
        self,
        persist_directory: Path = config.persist_directory,
        collection_name: str = config.collection_name,
        embedding_fn: Optional[SentenceTransformerEmbeddingFunction] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn or get_default_embedding_function()

        logger.info(f"Connecting to ChromaDB at {self.persist_directory}...")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Chroma collection '{self.collection_name}' ready. Current count: {self.collection.count()}")

    def add_parent_chunks(self, parent_chunks: List[ParentChunk]) -> int:
        """Flattens child chunks from parents and stores them in Chroma with parent metadata."""
        child_chunks: List[ChildChunk] = []
        for p in parent_chunks:
            child_chunks.extend(p.children)

        if not child_chunks:
            logger.warning("No child chunks to store.")
            return 0

        ids = [c.child_id for c in child_chunks]
        documents = [c.content for c in child_chunks]
        metadatas = [c.metadata for c in child_chunks]

        # Chroma handles embedding automatically via embedding_function
        # Process in batches of 64
        batch_size = 64
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )
            logger.info(f"Upserted {end_idx - i} child chunks to Chroma ({end_idx}/{len(ids)}).")

        logger.info(f"Successfully saved {len(child_chunks)} child chunks to Chroma.")
        return len(child_chunks)

    def dense_search(self, query: str, top_k: int = config.dense_top_k) -> List[Dict[str, Any]]:
        """Performs cosine similarity search for top-k child chunks in Chroma."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[f"search_query: {query}" if "nomic" in config.embedding_model_name else query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, child_id in enumerate(results["ids"][0]):
                doc_text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # Cosine similarity = 1 - cosine distance
                score = 1.0 - distance
                hits.append({
                    "child_id": child_id,
                    "content": doc_text,
                    "parent_id": metadata.get("parent_id", ""),
                    "parent_content": metadata.get("parent_content", ""),
                    "source": metadata.get("source", ""),
                    "score": score,
                    "retrieval_type": "dense",
                    "metadata": metadata,
                })
        return hits

    def get_all_child_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all child chunks from the collection for building the in-memory BM25 index."""
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(include=["documents", "metadatas"])
        chunks: List[Dict[str, Any]] = []
        for i, child_id in enumerate(data["ids"]):
            doc_text = data["documents"][i] if data["documents"] else ""
            metadata = data["metadatas"][i] if data["metadatas"] else {}
            chunks.append({
                "child_id": child_id,
                "content": doc_text,
                "parent_id": metadata.get("parent_id", ""),
                "parent_content": metadata.get("parent_content", ""),
                "source": metadata.get("source", ""),
                "metadata": metadata,
            })
        return chunks

    def clear(self) -> None:
        """Clears all records in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared Chroma collection '{self.collection_name}'.")
