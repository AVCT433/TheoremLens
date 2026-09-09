"""High-level Math RAG Pipeline orchestrating Ingestion, Storage, Retrieval, and Generation."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config import config
from ingestion import MathDocumentParser, ParentChildChunker, ParentChunk
from llm import OllamaMathLLM
from retriever import HybridRetriever, HybridSearchResult
from vector_store import ChromaStore

logger = logging.getLogger(__name__)


class MathRAGPipeline:
    """Orchestrates the end-to-end Local Math RAG pipeline:

    Ingestion (Docling) -> Storage (Chroma + BM25) -> Hybrid Retrieval & Guardrail -> LLM (Ollama)
    """

    def __init__(self):
        logger.info("Initializing MathRAGPipeline...")
        self.parser = MathDocumentParser()
        self.chunker = ParentChildChunker()
        self.vector_store = ChromaStore()
        self.retriever = HybridRetriever(vector_store=self.vector_store)
        self.llm = OllamaMathLLM()
        self.current_source: Optional[str] = None
        self.current_markdown: Optional[str] = None

    def ingest_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Parses PDF with Docling, creates parent-child chunks, saves to Chroma, and rebuilds BM25."""
        start_time = time.time()
        path = Path(pdf_path)
        logger.info(f"Starting ingestion for: {path.name}")

        # 1. Parse PDF using Docling
        markdown_text, source_name = self.parser.parse_to_markdown(path)
        self.current_source = source_name
        self.current_markdown = markdown_text

        # 2. Chunk text into Parent-Child hierarchy
        parents = self.chunker.split_document(markdown_text, source=source_name)
        total_children = sum(len(p.children) for p in parents)

        # 3. Store child chunks with parent metadata into Chroma
        inserted_count = self.vector_store.add_parent_chunks(parents)

        # 4. Rebuild in-memory BM25 index
        self.retriever.rebuild_bm25_index()

        elapsed = time.time() - start_time
        logger.info(f"Ingestion completed in {elapsed:.2f}s ({len(parents)} parents, {inserted_count} children).")

        return {
            "source": source_name,
            "parent_count": len(parents),
            "child_count": inserted_count,
            "markdown_preview": markdown_text[:800] + ("..." if len(markdown_text) > 800 else ""),
            "elapsed_seconds": elapsed,
        }

    def ingest_markdown_text(self, markdown_text: str, source_name: str = "custom_document.md") -> Dict[str, Any]:
        """Ingests raw markdown text directly (useful for testing or direct text input)."""
        start_time = time.time()
        self.current_source = source_name
        self.current_markdown = markdown_text

        parents = self.chunker.split_document(markdown_text, source=source_name)
        inserted_count = self.vector_store.add_parent_chunks(parents)
        self.retriever.rebuild_bm25_index()
        elapsed = time.time() - start_time

        return {
            "source": source_name,
            "parent_count": len(parents),
            "child_count": inserted_count,
            "markdown_preview": markdown_text[:800],
            "elapsed_seconds": elapsed,
        }

    def clear_database(self) -> None:
        """Clears Chroma storage and resets BM25 index."""
        self.vector_store.clear()
        self.retriever.rebuild_bm25_index()
        self.current_source = None
        self.current_markdown = None

    def query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Generator[str, None, None], HybridSearchResult]:
        """Executes retrieval, evaluates guardrail, and returns a generator streaming LLM tokens."""
        # Step 1: Hybrid Retrieval & CRAG Guardrail
        search_result = self.retriever.retrieve(question)

        # Step 2: If guardrail fails (max_score < 0.35), early-terminate without LLM call
        if not search_result.passed_guardrail:
            def early_terminate_stream():
                msg = search_result.guardrail_message or "문서에서 질문과 관련된 수학적 근거를 찾을 수 없습니다."
                yield f"🛡️ **[가드레일 조기 종료 (CRAG)]**\n\n{msg}\n\n"
                yield "- 최고 리랭킹 점수가 임계값(0.35) 미만이므로 환각 방지를 위해 LLM 추론을 생략했습니다."

            return early_terminate_stream(), search_result

        # Step 3: LLM generation streaming with retrieved parents
        llm_stream = self.llm.generate_stream(
            query=question,
            parents=search_result.top_parents,
            history=history,
        )
        return llm_stream, search_result
