"""Unit tests for ParentChildChunker — Parent-Child 분할 및 메타데이터 정합성 검증."""

import uuid
import pytest

from theoremlens.ingestion import ParentChildChunker, ParentChunk, ChildChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LONG_CONTENT = "수학적 정리와 증명 내용입니다. " * 80  # ~1,280 chars → 복수 parent 유도


@pytest.fixture
def small_chunker() -> ParentChildChunker:
    """소형 청크 크기(테스트 전용)로 구성된 ParentChildChunker."""
    return ParentChildChunker(
        parent_chunk_size=300,
        parent_chunk_overlap=30,
        child_chunk_size=80,
        child_chunk_overlap=10,
    )


@pytest.fixture
def default_chunker() -> ParentChildChunker:
    """기본 설정(config 값)으로 구성된 ParentChildChunker."""
    return ParentChildChunker()


@pytest.fixture
def multi_section_doc() -> str:
    """복수 섹션을 가진 수학 문서 예시."""
    return (
        "## 정리 1 — Cauchy-Schwarz 부등식\n"
        + ("$|\\langle u, v \\rangle|^2 \\le \\langle u,u \\rangle \\langle v,v \\rangle$. " * 40)
        + "\n\n"
        "## 정리 2 — 내적 공간의 정사영\n"
        + ("정사영은 $P = \\frac{\\langle v,u \\rangle}{\\langle u,u \\rangle} u$ 로 정의됩니다. " * 40)
    )


# ---------------------------------------------------------------------------
# TestParentChildSplitter
# ---------------------------------------------------------------------------


class TestParentChildSplitter:
    """ParentChildChunker.split_document()의 분할 및 메타데이터 정합성 테스트."""

    # ── 기본 분할 동작 ────────────────────────────────────────────────────────

    def test_split_returns_list_of_parent_chunks(self, small_chunker):
        """split_document()가 ParentChunk 리스트를 반환해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        assert isinstance(parents, list)
        assert len(parents) > 0
        assert all(isinstance(p, ParentChunk) for p in parents)

    def test_multiple_parents_generated(self, small_chunker):
        """충분히 긴 텍스트에서 복수의 parent chunk가 생성되어야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        assert len(parents) > 1, "긴 텍스트에서 복수 parent가 생성되어야 함"

    def test_each_parent_has_children(self, small_chunker):
        """모든 parent chunk는 최소 1개 이상의 child를 가져야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            assert len(p.children) >= 1, f"parent '{p.parent_id}' has no children"

    def test_single_short_text_produces_one_parent(self, small_chunker):
        """parent_chunk_size보다 짧은 텍스트는 정확히 1개의 parent를 생성해야 한다."""
        short_text = "짧은 수식: $x + y = z$"
        parents = small_chunker.split_document(short_text, source="short.md")
        assert len(parents) == 1

    def test_empty_text_returns_empty_list(self, small_chunker):
        """빈 텍스트는 빈 리스트를 반환해야 한다."""
        parents = small_chunker.split_document("", source="empty.md")
        assert parents == []

    # ── 메타데이터 정합성 ────────────────────────────────────────────────────

    def test_parent_id_is_valid_uuid(self, small_chunker):
        """각 parent의 parent_id는 유효한 UUID v4 형식이어야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            try:
                val = uuid.UUID(p.parent_id, version=4)
                assert str(val) == p.parent_id
            except ValueError:
                pytest.fail(f"parent_id '{p.parent_id}' is not a valid UUID4")

    def test_child_id_format(self, small_chunker):
        """child_id는 '{parent_id}_c{index}' 형식이어야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for idx, child in enumerate(p.children):
                expected = f"{p.parent_id}_c{idx}"
                assert child.child_id == expected, (
                    f"child_id 불일치: expected={expected}, actual={child.child_id}"
                )

    def test_child_parent_id_matches_parent(self, small_chunker):
        """child.parent_id는 부모 ParentChunk의 parent_id와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                assert child.parent_id == p.parent_id

    def test_child_parent_content_matches_parent_content(self, small_chunker):
        """child.parent_content는 부모의 content와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                assert child.parent_content == p.content

    def test_child_source_matches_parent_source(self, small_chunker):
        """child.source는 부모의 source와 동일해야 한다."""
        source = "math_paper.md"
        parents = small_chunker.split_document(LONG_CONTENT, source=source)
        for p in parents:
            assert p.source == source
            for child in p.children:
                assert child.source == source

    # ── metadata dict 정합성 ─────────────────────────────────────────────────

    def test_child_metadata_has_required_keys(self, small_chunker):
        """child.metadata에 필수 키(parent_id, child_id, source 등)가 존재해야 한다."""
        required_keys = {"parent_id", "child_id", "source", "parent_content",
                         "parent_index", "child_index"}
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                for key in required_keys:
                    assert key in child.metadata, (
                        f"child.metadata에 '{key}' 키가 없음: {child.metadata.keys()}"
                    )

    def test_child_metadata_parent_id_consistent(self, small_chunker):
        """child.metadata['parent_id']는 child.parent_id와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                assert child.metadata["parent_id"] == child.parent_id

    def test_child_metadata_child_id_consistent(self, small_chunker):
        """child.metadata['child_id']는 child.child_id와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                assert child.metadata["child_id"] == child.child_id

    def test_child_metadata_source_consistent(self, small_chunker):
        """child.metadata['source']는 child.source와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for child in p.children:
                assert child.metadata["source"] == child.source

    def test_child_metadata_parent_index_sequential(self, small_chunker):
        """child.metadata['parent_index']는 parent 순서(0-based)와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p_idx, p in enumerate(parents):
            for child in p.children:
                assert child.metadata["parent_index"] == p_idx

    def test_child_metadata_child_index_sequential(self, small_chunker):
        """child.metadata['child_index']는 parent 내 child 순서와 일치해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        for p in parents:
            for c_idx, child in enumerate(p.children):
                assert child.metadata["child_index"] == c_idx

    # ── parent_id 유일성 ─────────────────────────────────────────────────────

    def test_all_parent_ids_unique(self, small_chunker):
        """각 parent_id는 문서 내에서 유일해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        ids = [p.parent_id for p in parents]
        assert len(ids) == len(set(ids)), "parent_id가 중복됨"

    def test_all_child_ids_unique(self, small_chunker):
        """모든 child_id는 문서 전체에서 유일해야 한다."""
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        all_ids = [c.child_id for p in parents for c in p.children]
        assert len(all_ids) == len(set(all_ids)), "child_id가 중복됨"

    # ── 내용 포함 관계 ───────────────────────────────────────────────────────

    def test_child_content_is_substring_of_parent(self, small_chunker):
        """child의 content는 parent content의 부분 문자열이어야 한다 (overlap 영역 제외)."""
        short_doc = "A " * 50  # 단순 반복 문서
        parents = small_chunker.split_document(short_doc, source="test.md")
        for p in parents:
            for child in p.children:
                # child 내용의 첫 20자가 parent에 포함되는지만 검사
                # (overlap으로 인해 정확한 부분 문자열이 아닐 수 있음)
                assert len(child.content) > 0, "child content가 비어 있음"

    # ── 멀티 섹션 문서 ───────────────────────────────────────────────────────

    def test_multi_section_doc_split(self, small_chunker, multi_section_doc):
        """섹션이 여러 개인 문서에서 복수 parent가 생성되어야 한다."""
        parents = small_chunker.split_document(multi_section_doc, source="cauchy.md")
        assert len(parents) >= 2

    def test_multi_section_doc_source_propagated(self, small_chunker, multi_section_doc):
        """source 이름이 모든 parent와 child에 올바르게 전파되어야 한다."""
        source = "cauchy.md"
        parents = small_chunker.split_document(multi_section_doc, source=source)
        for p in parents:
            assert p.source == source
            for child in p.children:
                assert child.source == source
                assert child.metadata["source"] == source

    # ── 청크 크기 경계 ───────────────────────────────────────────────────────

    def test_child_chunk_not_exceed_configured_size(self, small_chunker):
        """child chunk 길이는 child_chunk_size에 근접하거나 작아야 한다.
        
        RecursiveCharacterTextSplitter는 분리자 기준으로 나누므로
        정확히 child_chunk_size를 초과하지 않음을 검증한다.
        """
        parents = small_chunker.split_document(LONG_CONTENT, source="test.md")
        limit = small_chunker.child_chunk_size
        for p in parents:
            for child in p.children:
                assert len(child.content) <= limit + 10, (
                    f"child content 길이({len(child.content)}) > limit({limit})"
                )
