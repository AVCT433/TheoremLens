"""
간단한 PDF 파싱 결과 출력 스크립트.

사용법:
    uv run python evals/parse_pdf.py <pdf_path>
    uv run python evals/parse_pdf.py <pdf_path> --output <output_path>
"""

import argparse
import sys
from pathlib import Path

# src 디렉터리를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from theoremlens.ingestion import MathDocumentParser

def main():
    parser = argparse.ArgumentParser(description="PDF 파일을 파싱하고 마크다운 결과를 출력합니다.")
    parser.add_argument("pdf_path", type=str, help="파싱할 PDF 파일 경로")
    parser.add_argument("-o", "--output", type=str, help="결과를 저장할 텍스트 파일 경로 (선택 사항)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ 오류: PDF 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
        
    print(f"📄 PDF 파싱 시작: {pdf_path.name}")
    print("⏳ Docling 초기화 및 파싱 중 (시간이 좀 소요될 수 있습니다)...")
    
    doc_parser = MathDocumentParser()
    try:
        markdown_text, source_name = doc_parser.parse_to_markdown(pdf_path)
    except Exception as e:
        print(f"\n❌ 파싱 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(markdown_text, encoding="utf-8")
        print(f"\n✅ 파싱 완료! 결과를 파일에 저장했습니다: {out_path}")
        print(f"   총 문자 수: {len(markdown_text):,}")
    else:
        print("\n" + "="*80)
        print(f"👇 파싱 결과 ({source_name}) 👇")
        print("="*80 + "\n")
        
        print(markdown_text)
        
        print("\n" + "="*80)
        print(f"✅ 파싱 완료! 총 문자 수: {len(markdown_text):,}")
        print("="*80)

if __name__ == "__main__":
    main()
