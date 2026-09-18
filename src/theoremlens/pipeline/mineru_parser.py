import subprocess
from pathlib import Path

def parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> Path:
    """
    MinerU CLI를 사용하여 PDF 파일을 파싱하고 생성된 JSON 결과물의 절대 경로를 반환합니다.
    
    Args:
        pdf_path (str): 파싱할 PDF 파일의 경로
        output_dir (str): 결과물이 저장될 디렉토리 경로
        
    Returns:
        Path: 생성된 '*_content_list.json' 파일의 절대 경로
        
    Raises:
        subprocess.CalledProcessError: MinerU 프로세스가 0이 아닌 상태 코드로 종료되었을 때
        FileNotFoundError: 결과물 JSON 파일을 찾지 못했을 때
    """
    # 명령어 리스트 구성
    cmd = [
        "uv", "run", "mineru",
        "-p", pdf_path,
        "-o", output_dir,
        "-b", "pipeline"
    ]
    
    try:
        # 서브프로세스 실행 (출력 캡처 및 에러 발생 시 예외 자동 발생 check=True)
        subprocess.run(
            cmd,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[MinerU Error] 명령어 실행 중 오류가 발생했습니다. (종료 코드: {e.returncode})")
        raise

    # output_dir을 Path 객체로 변환
    out_path = Path(output_dir)
    
    # output_dir 내부에서 패턴에 일치하는 JSON 파일 재귀적 탐색
    json_files = list(out_path.rglob("*_content_list.json"))
    
    if not json_files:
        raise FileNotFoundError(
            f"MinerU 작업이 완료되었으나 '{output_dir}' 내에서 "
            f"*_content_list.json 파일을 찾을 수 없습니다."
        )
        
    return json_files[0].resolve()

if __name__ == "__main__":
    # 1. 경로 설정
    test_pdf_path = r"C:\Users\baoro\Downloads\Enderton_Elements of set theory_(1977)-11-13.pdf"
    test_output_dir = "tests/results/test_output"

    # PDF 파일이 존재하는지 사전 체크
    if not Path(test_pdf_path).exists():
        print(f"에러: {test_pdf_path} 파일이 없습니다. 테스트용 PDF를 준비해주세요.")
        exit(1)

    print(f"MinerU 파싱 테스트 시작: {test_pdf_path}")
    
    # 2. 함수 호출
    try:
        result_json_path = parse_pdf_with_mineru(test_pdf_path, test_output_dir)
        print(f"\n테스트 성공! JSON 파일이 생성되었습니다.")
        print(f"절대 경로: {result_json_path}")
        
        # 3. 파일 크기 확인 (내용이 정상적으로 쓰였는지 검증)
        file_size = result_json_path.stat().st_size
        print(f"파일 크기: {file_size} bytes")
        
    except Exception as e:
        print(f"\n테스트 실패: {e}")