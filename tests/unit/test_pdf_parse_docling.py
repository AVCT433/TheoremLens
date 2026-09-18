"""
PDF 파싱 유닛 테스트.

사용법:
    # pytest로 실행 (PDF_PATH 환경변수 설정 필요)
    PDF_PATH=path/to/file.pdf uv run pytest tests/unit/test_pdf_parse.py -v -s

    # Windows PowerShell
    $env:PDF_PATH="path/to/file.pdf"; uv run pytest tests/unit/test_pdf_parse.py -v -s

    # 스크립트 직접 실행
    uv run python tests/unit/test_pdf_parse.py --pdf path/to/file.pdf
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 로그를 콘솔에 출력
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# src 패키지 경로 추가 (pytest 없이 직접 실행 시 필요)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from theoremlens.ingestion import MathDocumentParser, ParentChildChunker


# ──────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────

def _resolve_pdf_path() -> Path:
    env_path = os.environ.get("PDF_PATH")
    if env_path:
        return Path(env_path)
    pytest.skip("PDF_PATH 환경변수가 설정되지 않았습니다.")


@pytest.fixture(scope="module")
def pdf_path() -> Path:
    return _resolve_pdf_path()


@pytest.fixture(scope="module")
def parse_result(pdf_path: Path):
    parser = MathDocumentParser()
    markdown_text, source_name = parser.parse_to_markdown(pdf_path)
    return markdown_text, source_name


# ──────────────────────────────────────────────
# 파싱 테스트
# ──────────────────────────────────────────────

class TestMathDocumentParser:

    def test_file_exists(self, pdf_path: Path):
        assert pdf_path.exists(), f"파일을 찾을 수 없습니다: {pdf_path}"
        assert pdf_path.suffix.lower() == ".pdf"
        print(f"\n✅ PDF 파일 확인: {pdf_path.name}  ({pdf_path.stat().st_size / 1024:.1f} KB)")

    def test_parse_returns_nonempty_markdown(self, parse_result):
        markdown_text, source_name = parse_result
        assert isinstance(markdown_text, str)
        assert len(markdown_text) > 0
        print(f"\n✅ 파싱 성공: {source_name}")
        print(f"   총 문자 수: {len(markdown_text):,}  /  총 줄 수: {len(markdown_text.splitlines()):,}")

    def test_parse_contains_text(self, parse_result):
        markdown_text, _ = parse_result
        alpha_count = sum(1 for c in markdown_text if c.isalpha())
        assert alpha_count > 100
        print(f"\n✅ 알파벳 문자 수: {alpha_count:,}")

    def test_markdown_preview(self, parse_result):
        markdown_text, source_name = parse_result
        lines = markdown_text.splitlines()
        preview = lines[:40]
        print(f"\n{'='*64}")
        print(f"📄  {source_name}  —  마크다운 미리보기 (앞 40줄 / 전체 {len(lines)}줄)")
        print(f"{'='*64}")
        for i, line in enumerate(preview, 1):
            print(f"{i:3d} │ {line}")
        if len(lines) > 40:
            print(f"    │ ... ({len(lines) - 40}줄 생략)")
        print(f"{'='*64}")


# ──────────────────────────────────────────────
# 청킹 테스트
# ──────────────────────────────────────────────

class TestParentChildChunker:

    @pytest.fixture(scope="class")
    def chunk_result(self, parse_result):
        markdown_text, source_name = parse_result
        chunker = ParentChildChunker()
        return chunker.split_document(markdown_text, source=source_name)

    def test_produces_parent_chunks(self, chunk_result):
        assert len(chunk_result) > 0
        print(f"\n✅ Parent 청크 수: {len(chunk_result)}")

    def test_each_parent_has_children(self, chunk_result):
        for p in chunk_result:
            assert len(p.children) > 0
        total = sum(len(p.children) for p in chunk_result)
        print(f"\n✅ 총 Child 청크 수: {total}")

    def test_child_ids_are_unique(self, chunk_result):
        ids = [c.child_id for p in chunk_result for c in p.children]
        assert len(ids) == len(set(ids))
        print(f"\n✅ Child ID 유일성 확인: {len(ids)}개 모두 고유")

    def test_chunk_summary(self, chunk_result):
        parents = chunk_result
        total_children = sum(len(p.children) for p in parents)
        avg_p = sum(len(p.content) for p in parents) / len(parents)
        avg_c = (
            sum(len(c.content) for p in parents for c in p.children) / total_children
            if total_children > 0 else 0
        )
        print(f"\n{'='*64}")
        print(f"📊  청킹 결과 요약")
        print(f"{'='*64}")
        print(f"  Parent 청크 수    : {len(parents)}")
        print(f"  Child 청크 수     : {total_children}")
        print(f"  평균 Parent 길이  : {avg_p:.0f} chars")
        print(f"  평균 Child 길이   : {avg_c:.0f} chars")
        print(f"{'─'*64}")
        first_p = parents[0]
        print(f"\n[Parent #0 미리보기 — {len(first_p.content)} chars]")
        print(first_p.content[:400])
        if first_p.children:
            print(f"\n[Child #0 미리보기 — {len(first_p.children[0].content)} chars]")
            print(first_p.children[0].content[:200])
        print(f"{'='*64}")


# ──────────────────────────────────────────────
# 스크립트 직접 실행 모드
# ──────────────────────────────────────────────

def _run_direct(pdf_path_str: str) -> None:
    pdf_path = Path(pdf_path_str)
    print(f"\n{'='*64}")
    print("🔍  PDF 파싱 테스트 (직접 실행)")
    print(f"{'='*64}")
    print(f"  대상 파일: {pdf_path}")
    if not pdf_path.exists():
        print(f"\n❌  파일을 찾을 수 없습니다 → {pdf_path}")
        sys.exit(1)
    print(f"  파일 크기: {pdf_path.stat().st_size / 1024:.1f} KB\n")

    print("▶  [1/2] PDF → Markdown 변환 중...")
    parser = MathDocumentParser()
    markdown_text, source_name = parser.parse_to_markdown(pdf_path)
    lines = markdown_text.splitlines()
    print(f"✅  변환 완료: {len(markdown_text):,} 문자  /  {len(lines):,} 줄\n")

    print(f"{'─'*64}")
    print(f"📄  마크다운 미리보기 (앞 40줄 / 전체 {len(lines)} 줄)")
    print(f"{'─'*64}")
    for i, line in enumerate(lines[:40], 1):
        print(f"{i:3d} │ {line}")
    if len(lines) > 40:
        print(f"    │ ... ({len(lines) - 40}줄 생략)")
    print(f"{'─'*64}\n")

    print("▶  [2/2] Parent-Child 청킹 중...")
    chunker = ParentChildChunker()
    parents = chunker.split_document(markdown_text, source=source_name)
    total_children = sum(len(p.children) for p in parents)
    print(f"✅  청킹 완료\n")

    avg_p = sum(len(p.content) for p in parents) / max(len(parents), 1)
    avg_c = sum(len(c.content) for p in parents for c in p.children) / max(total_children, 1)
    print(f"{'─'*64}")
    print(f"📊  청킹 결과 요약")
    print(f"{'─'*64}")
    print(f"  Parent 청크 수    : {len(parents)}")
    print(f"  Child 청크 수     : {total_children}")
    print(f"  평균 Parent 길이  : {avg_p:.0f} chars")
    print(f"  평균 Child 길이   : {avg_c:.0f} chars")
    print(f"{'─'*64}")

    if parents:
        first_p = parents[0]
        print(f"\n[Parent #0 미리보기 — {len(first_p.content)} chars]")
        print(first_p.content[:400])
        if first_p.children:
            print(f"\n[Child #0 미리보기 — {len(first_p.children[0].content)} chars]")
            print(first_p.children[0].content[:200])

    print(f"\n{'='*64}")
    print("✅  완료")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="PDF 파싱 테스트")
    arg_parser.add_argument("--pdf", required=True, help="파싱할 PDF 파일 경로")
    args = arg_parser.parse_args()
    _run_direct(args.pdf)
