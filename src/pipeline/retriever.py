"""Hybrid retrieval (Dense + Sparse BM25 with LaTeX preservation), Cross-Encoder reranking, and CRAG guardrail."""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import config
from src.pipeline.vector_store import ChromaStore

logger = logging.getLogger(__name__)


def latex_regex_tokenizer(text: str, pattern: str = config.bm25_token_pattern) -> List[str]:
    """Tokenizes text preserving LaTeX commands (e.g. \\frac, \\sum, \\alpha) and mathematical symbols."""
    if not text:
        return []
    # Tokenize preserving backslashed commands, alphanumerics, and individual symbols
    tokens = re.findall(pattern, text)
    # Lowercase tokens except LaTeX commands that might be case-sensitive or standard
    return [t if t.startswith("\\") else t.lower() for t in tokens]


@dataclass
class RetrievedParent:
    parent_id: str
    parent_content: str
    source: str
    max_child_score: float
    matched_children: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HybridSearchResult:
    passed_guardrail: bool
    top_parents: List[RetrievedParent]
    max_reranker_score: float
    dense_hits: List[Dict[str, Any]]
    sparse_hits: List[Dict[str, Any]]
    union_candidates: List[Dict[str, Any]]
    reranked_candidates: List[Dict[str, Any]]
    guardrail_message: Optional[str] = None


class LuceneBM25(BM25Okapi):
    """BM25 implementation subclassing BM25Okapi with Apache Lucene's IDF formula:
    IDF = ln(1 + (N - n + 0.5) / (n + 0.5))

    This guarantees strictly positive IDF scores for all terms regardless of corpus size.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            # Apache Lucene standard BM25 IDF formula
            idf = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf


class InMemoryBM25Index:
    """In-memory BM25 index built dynamically from stored child chunks in Chroma."""

    def __init__(self, token_pattern: str = config.bm25_token_pattern):
        self.token_pattern = token_pattern
        self.bm25: Optional[LuceneBM25] = None
        self.corpus_chunks: List[Dict[str, Any]] = []

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Builds BM25 index in memory (~0.02s)."""
        self.corpus_chunks = chunks
        if not chunks:
            logger.warning("BM25: No chunks to index.")
            self.bm25 = None
            return

        tokenized_corpus = [
            latex_regex_tokenizer(chunk["content"], self.token_pattern)
            for chunk in chunks
        ]
        self.bm25 = LuceneBM25(tokenized_corpus)
        logger.info(f"Built in-memory BM25 index for {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = config.sparse_top_k) -> List[Dict[str, Any]]:
        """Searches BM25 index and returns top-k hits."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = latex_regex_tokenizer(query, self.token_pattern)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        hits: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue
            chunk = self.corpus_chunks[idx]
            hits.append({
                "child_id": chunk["child_id"],
                "content": chunk["content"],
                "parent_id": chunk["parent_id"],
                "parent_content": chunk["parent_content"],
                "source": chunk["source"],
                "score": score,
                "retrieval_type": "sparse",
                "metadata": chunk.get("metadata", {}),
            })
        return hits


class CrossEncoderReranker:
    """Cross-Encoder reranker using BAAI/bge-reranker-base."""

    def __init__(
        self,
        model_name: str = config.reranker_model_name,
        device: str = config.reranker_device,
        threshold: float = config.reranker_threshold,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading Cross-Encoder reranker '{self.model_name}' on {self.device.upper()}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Cross-Encoder reranker loaded successfully.")
        return self._model

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scores (query, candidate) pairs using BAAI/bge-reranker-base.

        CrossEncoder의 default activation_fn이 Sigmoid이므로,
        predict() 반환값은 이미 [0, 1] 범위로 정규화된 점수입니다.
        """
        if not candidates:
            return []

        pairs = [[query, cand["content"]] for cand in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for i, score in enumerate(scores):
            cand_copy = dict(candidates[i])
            cand_copy["reranker_score"] = float(score)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
        return scored_candidates


class HybridRetriever:
    """Orchestrates Dense (Chroma) + Sparse (BM25) search, Union, BGE reranking, and CRAG guardrails."""

    def __init__(
        self,
        vector_store: ChromaStore,
        reranker: Optional[CrossEncoderReranker] = None,
        dense_top_k: int = config.dense_top_k,
        sparse_top_k: int = config.sparse_top_k,
        final_top_parents: int = config.final_top_parents,
        threshold: float = config.reranker_threshold,
    ):
        self.vector_store = vector_store
        self.reranker = reranker or CrossEncoderReranker(threshold=threshold)
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_parents = final_top_parents
        self.threshold = threshold

        self.bm25_index = InMemoryBM25Index()
        self.rebuild_bm25_index()

    def rebuild_bm25_index(self) -> None:
        """Fetches all chunks from Chroma and rebuilds in-memory BM25 index."""
        chunks = self.vector_store.get_all_child_chunks()
        self.bm25_index.build_from_chunks(chunks)

    def retrieve(self, query: str) -> HybridSearchResult:
        """Executes the full retrieval and guardrail pipeline:

        1. Dense Top-10 (Chroma) + Sparse Top-10 (BM25)
        2. Union (deduplicated candidates)
        3. Cross-Encoder rerank (bge-reranker-base on CPU)
        4. Guardrail cutoff check (threshold = 0.35)
        5. Parent de-duplication & retrieval of top 1~2 parent contents
        """
        # 1. First-stage hybrid retrieval
        dense_hits = self.vector_store.dense_search(query, top_k=self.dense_top_k)
        sparse_hits = self.bm25_index.search(query, top_k=self.sparse_top_k)

        # 2. Simple Union Strategy (deduplicate by child_id)
        seen_child_ids: Set[str] = set()
        union_candidates: List[Dict[str, Any]] = []

        for hit in dense_hits + sparse_hits:
            c_id = hit["child_id"]
            if c_id not in seen_child_ids:
                seen_child_ids.add(c_id)
                union_candidates.append(hit)

        if not union_candidates:
            logger.info("Retrieval: No candidates found from either Dense or Sparse search.")
            return HybridSearchResult(
                passed_guardrail=False,
                top_parents=[],
                max_reranker_score=0.0,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                union_candidates=[],
                reranked_candidates=[],
                guardrail_message="문서에서 관련 있는 내용을 찾을 수 없습니다.",
            )

        # 3. Cross-Encoder reranking
        reranked = self.reranker.rerank(query, union_candidates)
        max_score = reranked[0]["reranker_score"] if reranked else 0.0

        # 4. Guardrail cutoff (CRAG early termination if max_score < threshold)
        if max_score < self.threshold:
            logger.warning(
                f"Guardrail triggered: Max reranker score ({max_score:.4f}) is below threshold ({self.threshold}). "
                "Early terminating LLM call to prevent hallucination."
            )
            return HybridSearchResult(
                passed_guardrail=False,
                top_parents=[],
                max_reranker_score=max_score,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                union_candidates=union_candidates,
                reranked_candidates=reranked,
                guardrail_message=(
                    f"문서에서 질문과 관련된 수학적 근거를 찾을 수 없습니다. "
                    f"(최고 관련도 점수: {max_score:.3f} < 기준치 {self.threshold:.2f})"
                ),
            )

        # 5. Parent de-duplication & top parent selection
        valid_candidates = [c for c in reranked if c["reranker_score"] >= self.threshold]
        seen_parent_ids: Set[str] = set()
        top_parents: List[RetrievedParent] = []

        for cand in valid_candidates:
            p_id = cand["parent_id"]
            if p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                top_parents.append(
                    RetrievedParent(
                        parent_id=p_id,
                        parent_content=cand["parent_content"],
                        source=cand["source"],
                        max_child_score=cand["reranker_score"],
                        matched_children=[cand],
                    )
                )
                if len(top_parents) >= self.final_top_parents:
                    break
            else:
                # Add child to existing parent info
                for p in top_parents:
                    if p.parent_id == p_id:
                        p.matched_children.append(cand)
                        break

        logger.info(
            f"Retrieval success: {len(top_parents)} parent chunks selected. Max score: {max_score:.4f}"
        )
        return HybridSearchResult(
            passed_guardrail=True,
            top_parents=top_parents,
            max_reranker_score=max_score,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            union_candidates=union_candidates,
            reranked_candidates=reranked,
        )
