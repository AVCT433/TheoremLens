"""Unit tests for InMemoryBM25Index — BM25 점수 산출 및 정렬 로직 검증."""

import pytest

from theoremlens.pipeline.retriever import InMemoryBM25Index, LuceneBM25


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_chunk(child_id: str, content: str, parent_id: str = "p1") -> dict:
    return {
        "child_id": child_id,
        "content": content,
        "parent_id": parent_id,
        "parent_content": f"parent of {child_id}",
        "source": "math.md",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MATH_CORPUS = [
    _make_chunk(
        "c1",
        r"Let $x \in V$ be a vector in an inner product space.",
        "p1",
    ),
    _make_chunk(
        "c2",
        r"We calculate \int_0^1 f(t) dt using integration by parts.",
        "p2",
    ),
    _make_chunk(
        "c3",
        r"The Cauchy-Schwarz inequality states $|\langle u,v\rangle|^2 \le \langle u,u\rangle\langle v,v\rangle$.",
        "p3",
    ),
    _make_chunk(
        "c4",
        r"Eigenvalues \lambda satisfy the characteristic equation \det(A - \lambda I) = 0.",
        "p4",
    ),
    _make_chunk(
        "c5",
        r"The gradient \nabla f points in the direction of steepest ascent.",
        "p5",
    ),
]


@pytest.fixture
def bm25_index() -> InMemoryBM25Index:
    """MATH_CORPUS로 빌드된 InMemoryBM25Index."""
    idx = InMemoryBM25Index()
    idx.build_from_chunks(MATH_CORPUS)
    return idx


@pytest.fixture
def empty_index() -> InMemoryBM25Index:
    """빈 코퍼스로 초기화된 InMemoryBM25Index."""
    return InMemoryBM25Index()


# ---------------------------------------------------------------------------
# TestBM25IndexBuild
# ---------------------------------------------------------------------------


class TestBM25IndexBuild:
    """build_from_chunks() 초기화 및 내부 상태 검증."""

    def test_build_sets_bm25_instance(self, bm25_index):
        """build_from_chunks() 후 내부 bm25 인스턴스가 생성되어야 한다."""
        assert bm25_index.bm25 is not None
        assert isinstance(bm25_index.bm25, LuceneBM25)

    def test_build_stores_corpus_chunks(self, bm25_index):
        """corpus_chunks가 입력 청크 개수만큼 저장되어야 한다."""
        assert len(bm25_index.corpus_chunks) == len(MATH_CORPUS)

    def test_build_empty_corpus_leaves_bm25_none(self, empty_index):
        """빈 청크 리스트로 build() 호출 시 bm25가 None 상태여야 한다."""
        empty_index.build_from_chunks([])
        assert empty_index.bm25 is None

    def test_rebuild_replaces_old_index(self, bm25_index):
        """build_from_chunks()를 다시 호출하면 이전 인덱스를 대체해야 한다."""
        new_corpus = [_make_chunk("n1", "totally different corpus about topology")]
        bm25_index.build_from_chunks(new_corpus)
        assert len(bm25_index.corpus_chunks) == 1


# ---------------------------------------------------------------------------
# TestBM25Search
# ---------------------------------------------------------------------------


class TestBM25Search:
    """search() 결과 구조, 점수 산출, 정렬 로직 검증."""

    # ── 기본 반환 구조 ────────────────────────────────────────────────────────

    def test_search_returns_list(self, bm25_index):
        """search()는 항상 list를 반환해야 한다."""
        result = bm25_index.search("inner product", top_k=3)
        assert isinstance(result, list)

    def test_search_hit_has_required_keys(self, bm25_index):
        """각 히트 dict에 필수 키가 포함되어야 한다."""
        required = {"child_id", "content", "parent_id", "parent_content",
                    "source", "score", "retrieval_type"}
        hits = bm25_index.search("inner product", top_k=3)
        for hit in hits:
            for key in required:
                assert key in hit, f"'{key}' 키가 히트에 없음"

    def test_retrieval_type_is_sparse(self, bm25_index):
        """모든 히트의 retrieval_type은 'sparse'이어야 한다."""
        hits = bm25_index.search("eigenvalue", top_k=5)
        for hit in hits:
            assert hit["retrieval_type"] == "sparse"

    def test_score_is_positive_float(self, bm25_index):
        """반환된 모든 히트의 score는 양수 float이어야 한다."""
        hits = bm25_index.search("inner product space", top_k=5)
        for hit in hits:
            assert isinstance(hit["score"], float)
            assert hit["score"] > 0.0

    # ── 관련성 순위 검증 ──────────────────────────────────────────────────────

    def test_most_relevant_chunk_ranked_first_plain_query(self, bm25_index):
        """'inner product' 쿼리에서 관련 청크(c1)가 1위여야 한다."""
        hits = bm25_index.search("inner product", top_k=5)
        assert len(hits) > 0
        assert hits[0]["child_id"] == "c1"

    def test_latex_command_query_ranked_correctly(self, bm25_index):
        r"""'\\int' LaTeX 커맨드 쿼리에서 적분 관련 청크(c2)가 1위여야 한다."""
        hits = bm25_index.search(r"\int", top_k=5)
        assert len(hits) > 0
        assert hits[0]["child_id"] == "c2"

    def test_eigenvalue_query_ranks_c4_first(self, bm25_index):
        r"""'eigenvalue lambda' 쿼리에서 c4 청크가 최상위여야 한다."""
        hits = bm25_index.search(r"eigenvalue \lambda", top_k=5)
        assert len(hits) > 0
        assert hits[0]["child_id"] == "c4"

    def test_gradient_query_ranks_c5_first(self, bm25_index):
        r"""'gradient direction' 쿼리에서 c5 청크가 최상위여야 한다."""
        hits = bm25_index.search(r"gradient \nabla direction", top_k=5)
        assert len(hits) > 0
        assert hits[0]["child_id"] == "c5"

    def test_results_sorted_by_score_descending(self, bm25_index):
        """반환 결과는 score 내림차순으로 정렬되어야 한다."""
        hits = bm25_index.search("vector space integration eigenvalue", top_k=5)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True), "score 내림차순 정렬 위반"

    # ── top_k 제한 검증 ───────────────────────────────────────────────────────

    def test_top_k_limits_results(self, bm25_index):
        """top_k=2이면 최대 2개의 결과만 반환되어야 한다."""
        hits = bm25_index.search("inner product vector", top_k=2)
        assert len(hits) <= 2

    def test_top_k_1_returns_single_best(self, bm25_index):
        """top_k=1이면 정확히 1개의 최적 결과만 반환되어야 한다."""
        hits = bm25_index.search("integration by parts", top_k=1)
        assert len(hits) == 1

    def test_top_k_larger_than_corpus(self, bm25_index):
        """top_k가 코퍼스 크기를 초과해도 오류 없이 동작해야 한다."""
        hits = bm25_index.search("vector", top_k=100)
        assert len(hits) <= len(MATH_CORPUS)

    # ── 엣지 케이스 ──────────────────────────────────────────────────────────

    def test_search_on_empty_index_returns_empty(self, empty_index):
        """인덱스가 빌드되지 않은 상태에서 search()는 빈 리스트를 반환해야 한다."""
        result = empty_index.search("anything", top_k=5)
        assert result == []

    def test_empty_query_returns_empty(self, bm25_index):
        """빈 문자열 쿼리는 빈 리스트를 반환해야 한다."""
        result = bm25_index.search("", top_k=5)
        assert result == []

    def test_query_with_no_match_returns_empty(self, bm25_index):
        """코퍼스에 전혀 없는 단어로 검색 시 빈 리스트를 반환해야 한다."""
        result = bm25_index.search("xyzzyquux123nonexistenttoken", top_k=5)
        assert result == []

    def test_child_id_in_result_exists_in_corpus(self, bm25_index):
        """검색 결과의 child_id는 원본 코퍼스에 존재해야 한다."""
        corpus_ids = {c["child_id"] for c in MATH_CORPUS}
        hits = bm25_index.search("vector", top_k=5)
        for hit in hits:
            assert hit["child_id"] in corpus_ids

    # ── Lucene IDF 공식 검증 ─────────────────────────────────────────────────

    def test_lucene_idf_scores_are_non_negative(self, bm25_index):
        """Lucene BM25 IDF 공식에 의해 모든 IDF 값이 비음수여야 한다."""
        bm25 = bm25_index.bm25
        assert bm25 is not None
        for word, idf_val in bm25.idf.items():
            assert idf_val >= 0.0, f"IDF of '{word}' is negative: {idf_val}"

    def test_rare_term_has_higher_idf_than_common_term(self):
        """희귀 단어의 IDF가 자주 등장하는 단어의 IDF보다 커야 한다."""
        # 'common' 5회, 'rare' 1회 등장
        corpus = [
            _make_chunk(f"c{i}", f"common word repeated again and again doc {i}")
            for i in range(5)
        ] + [_make_chunk("cr", "rare unique xyzzy term only here")]

        idx = InMemoryBM25Index()
        idx.build_from_chunks(corpus)

        bm25 = idx.bm25
        assert bm25 is not None
        common_idf = bm25.idf.get("common", 0.0)
        rare_idf = bm25.idf.get("xyzzy", 0.0)
        assert rare_idf > common_idf, (
            f"rare IDF({rare_idf:.4f}) should be > common IDF({common_idf:.4f})"
        )

    # ── 메타데이터 전달 검증 ─────────────────────────────────────────────────

    def test_hit_metadata_field_preserved(self):
        """청크에 metadata 필드가 있으면 히트에 그대로 전달되어야 한다."""
        chunk_with_meta = _make_chunk("cm1", "Hilbert space is a complete inner product space")
        chunk_with_meta["metadata"] = {"chapter": 3, "page": 42}

        idx = InMemoryBM25Index()
        idx.build_from_chunks([chunk_with_meta])
        hits = idx.search("Hilbert space", top_k=1)

        assert len(hits) == 1
        assert hits[0]["metadata"]["chapter"] == 3
        assert hits[0]["metadata"]["page"] == 42

    def test_chunk_without_metadata_uses_empty_dict(self):
        """metadata 키가 없는 청크는 히트의 metadata로 빈 dict를 반환해야 한다."""
        chunk_no_meta = _make_chunk("cnm1", "metric space is a generalization of normed space")
        # metadata 키 제거
        chunk_no_meta.pop("metadata", None)

        idx = InMemoryBM25Index()
        idx.build_from_chunks([chunk_no_meta])
        hits = idx.search("metric space", top_k=1)

        assert len(hits) == 1
        assert hits[0]["metadata"] == {}
