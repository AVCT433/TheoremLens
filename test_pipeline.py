"""Unit and integration tests for TheoremLens components."""

import unittest
from config import config
from ingestion import ParentChildChunker
from retriever import (
    CrossEncoderReranker,
    InMemoryBM25Index,
    latex_regex_tokenizer,
)
from vector_store import ChromaStore


class TestTheoremLens(unittest.TestCase):

    def test_latex_tokenizer_preserves_commands(self):
        sample_latex = r"Here is \frac{a}{b} and \alpha + \beta = \gamma and x^2 \le 10"
        tokens = latex_regex_tokenizer(sample_latex)
        self.assertIn(r"\frac", tokens)
        self.assertIn(r"\alpha", tokens)
        self.assertIn(r"\beta", tokens)
        self.assertIn(r"\gamma", tokens)
        self.assertIn(r"\le", tokens)

    def test_parent_child_chunker(self):
        sample_doc = (
            "## 정리 1\n" + ("수학적 내용입니다. " * 80) + "\n\n"
            "## 정리 2\n" + ("다른 증명 내용입니다. " * 80)
        )
        chunker = ParentChildChunker(
            parent_chunk_size=500,
            parent_chunk_overlap=50,
            child_chunk_size=120,
            child_chunk_overlap=20,
        )
        parents = chunker.split_document(sample_doc, source="test.md")
        self.assertGreater(len(parents), 1)

        # Verify parent-child relationship
        for p in parents:
            self.assertTrue(p.parent_id)
            self.assertGreater(len(p.children), 0)
            for c in p.children:
                self.assertEqual(c.parent_id, p.parent_id)
                self.assertEqual(c.parent_content, p.content)
                self.assertEqual(c.metadata["parent_id"], p.parent_id)

    def test_bm25_in_memory_index(self):
        chunks = [
            {
                "child_id": "c1",
                "content": r"Let $x \in V$ be a vector in an inner product space.",
                "parent_id": "p1",
                "parent_content": "Full parent 1",
                "source": "math.md",
            },
            {
                "child_id": "c2",
                "content": r"We calculate \int_0^1 f(t) dt using integration by parts.",
                "parent_id": "p2",
                "parent_content": "Full parent 2",
                "source": "math.md",
            },
        ]
        bm25_idx = InMemoryBM25Index()
        bm25_idx.build_from_chunks(chunks)

        hits = bm25_idx.search("inner product", top_k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0]["child_id"], "c1")

        latex_hits = bm25_idx.search(r"\int", top_k=2)
        self.assertGreater(len(latex_hits), 0)
        self.assertEqual(latex_hits[0]["child_id"], "c2")


if __name__ == "__main__":
    unittest.main()
