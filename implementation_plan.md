# Local Math RAG Architecture (MVP v1.1) 구현 계획

학술 수학 PDF를 대상으로 LaTeX 수식 훼손과 환각을 원천 차단하는 고정밀 로컬 RAG 파이프라인 및 Gradio 웹 애플리케이션을 구축합니다.

---

## 1. 개요 및 하드웨어 분업 전략

* **GPU (RTX 2060 Super 8GB VRAM)**:
  * LLM 전담: `qwen2.5-coder:7b` (Ollama, `num_ctx: 8192`)
  * VRAM 스왑 방지를 위해 임베딩/리랭커를 GPU에서 완전 배제
* **CPU / 시스템 RAM**:
  * 문서 파싱: `Docling` (경량 비전/ONNX 기반, 수식 Markdown 추출)
  * 임베딩: `nomic-embed-text` (`device="cpu"`, 768차원)
  * 리랭커: `bge-reranker-base` (Cross-Encoder, `device="cpu"`)
  * BM25: `rank-bm25` (LaTeX 보존 정규식 토크나이저, 인메모리)

---

## 2. User Review Required (사용자 사전 확인 필요 사항)

> [!IMPORTANT]
> 1. **Ollama 서비스 및 모델**: 로컬에 Ollama가 실행 중이고 `qwen2.5-coder:7b` 모델이 다운로드(`ollama pull qwen2.5-coder:7b`)되어 있는지 확인이 필요합니다.
> 2. **임베딩 방식**: `nomic-embed-text`는 HuggingFace/SentenceTransformers를 통해 CPU에서 완전히 독립 구동되도록 구성하여 Ollama의 VRAM 점유를 방지합니다. (Ollama API 임베딩도 옵션으로 선택 가능)
> 3. **리랭커 점수 컷오프**: `bge-reranker-base`의 raw logit에 Sigmoid를 취해 0~1 스케일의 점수를 계산하며, 스펙에 따라 `threshold = 0.35` 미만일 경우 LLM 호출을 즉시 생략하고 조기 종료합니다.

---

## 3. 모듈별 상세 설계 및 Proposed Changes

### Configuration Layer
#### [NEW] [config.py](file:///d:/Projects/TheoremLens/config.py)
* 청크 파라미터 (부모: 1200자/150 overlap, 자식: 250자/40 overlap)
* 구분자 우선순위 (`\n## `, `\n### `, `\n\n`, `\n`, ` `)
* 하이브리드 검색 Top-K (Dense: 10, Sparse: 10)
* 리랭커 임계값 (`threshold = 0.35`) 및 최종 상위 부모 청크 수 (`top_parents = 2`)
* Ollama 설정 (`base_url="http://localhost:11434"`, `model="qwen2.5-coder:7b"`, `num_ctx=8192`)
* 디렉토리 경로 설정 (`chroma_db`, `uploads` 등)

---

### Ingestion & Chunking Layer
#### [NEW] [ingestion.py](file:///d:/Projects/TheoremLens/ingestion.py)
* `Docling`의 `DocumentConverter`를 사용하여 PDF를 파싱하고 `export_to_markdown()`으로 수식(`$...$`, `$$...$$`) 및 표를 보존한 텍스트 추출
* `RecursiveCharacterTextSplitter`를 활용한 **Parent-Child Chunker**:
  1. 부모 분할: 1200자 (Overlap 150자)
  2. 부모 청크마다 고유 `parent_id` (UUID) 부여
  3. 각 부모 청크 내부를 자식 청크 250자 (Overlap 40자)로 분할
  4. 자식 청크 메타데이터에 `parent_id`, `parent_content`, `source` 주입

---

### Storage & Embedding Layer
#### [NEW] [vector_store.py](file:///d:/Projects/TheoremLens/vector_store.py)
* Chroma 로컬 영속화 관리 (`persist_directory="./chroma_db"`)
* `nomic-embed-text` CPU 전용 임베딩 함수 구현 (`device="cpu"`, 768차원)
* 자식 청크 배치 삽입 및 조회 유틸리티
* 기존 컬렉션에서 전체 자식 청크 데이터를 읽어와 BM25 인덱스 빌더로 전달하는 기능

---

### Retrieval & Guardrail Layer
#### [NEW] [retriever.py](file:///d:/Projects/TheoremLens/retriever.py)
* **Sparse (BM25)**:
  * LaTeX 명령어 보존 토크나이저: `re.findall(r"\\[a-zA-Z]+|[a-zA-Z0-9]+|[^\s\w]", text)`
  * 앱 구동/인제스천 시 인메모리 즉석 빌드
  * BM25 Top-10 검색
* **Dense (Chroma)**:
  * Chroma 코사인 유사도 기반 자식 청크 Top-10 검색
* **후보 결합 (Union)**:
  * Dense Top-10 + BM25 Top-10 단순 합집합 (중복 제거 후 약 14~18개 청크)
* **리랭킹 및 조기 종료 (CRAG Guardrail)**:
  * `BAAI/bge-reranker-base` (CrossEncoder, `device="cpu"`)
  * 각 후보 자식 청크에 대해 점수 산출 (Sigmoid 변환)
  * `max_score < 0.35`인 경우: 관련 근거 없음 판정 -> 조기 종료 트리거
* **부모 역추적 (De-duplication)**:
  * 통과한 자식 청크들의 `parent_id`를 기준으로 중복을 제거하여 상위 1~2개 부모 청크 원문 추출

---

### Generation & Prompt Layer
#### [NEW] [llm.py](file:///d:/Projects/TheoremLens/llm.py)
* Ollama API 클라이언트 (`qwen2.5-coder:7b`, `num_ctx: 8192`, `temperature: 0.0`)
* **엄격한 폐쇄형 RAG 시스템 프롬프트**:
  * 제공된 문서 컨텍스트 외 자의적 수학적 가정/유도 생성 금지
  * 수식 작성 시 인라인 `$...$`, 블록 `$$...$$` LaTeX 문법 강제
  * 각 진술 문장 끝에 출처 태깅 의무화 (`[출처: 부모 청크 #1]` 등)
* 비동기/동기 스트리밍 생성 제너레이터 구현

---

### Pipeline Orchestration
#### [NEW] [pipeline.py](file:///d:/Projects/TheoremLens/pipeline.py)
* `MathRAGPipeline` 클래스:
  * `ingest_pdf(file_path)`: PDF 파싱 -> 청킹 -> Chroma 저장 -> BM25 인덱스 리빌드
  * `query(question, stream=True)`: Retrieval -> Guardrail 확인 -> LLM 생성 (스트리밍)
  * 디버그 정보 반환 (Dense/BM25 결과, 리랭킹 점수, 조기 종료 여부, 선택된 부모 컨텍스트)

---

### Web User Interface
#### [NEW] [app.py](file:///d:/Projects/TheoremLens/app.py)
* `Gradio` 기반 직관적이고 미려한 웹 인터페이스:
  * `latex_delimiters=[{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]` 적용
  * 좌측 사이드바: PDF 업로드, 인제스천 진행 상태, 인덱싱된 청크 통계, 하이퍼파라미터(임계값, num_ctx) 조절
  * 우측 메인 영역:
    * 수식 완벽 렌더링 스트리밍 대화창
    * 검색 인스펙터(Accordion): Dense/BM25 검색 결과, 리랭커 점수, 가드레일 상태, 주입된 부모 청크 원문 실시간 확인

---

### Evaluation & Verification
#### [NEW] [eval.py](file:///d:/Projects/TheoremLens/eval.py)
* Golden Dataset 기반 Retrieval Hit Rate 및 0.35 컷오프 가드레일 동작 검증 스크립트

#### [NEW] [pyproject.toml](file:///d:/Projects/TheoremLens/pyproject.toml) / [README.md](file:///d:/Projects/TheoremLens/README.md)
* `uv` 환경 프로젝트 설정 및 실행 가이드 문서 작성

---

## 4. Verification Plan

### Automated Verification
1. **단위 테스트**:
   * 토크나이저 정규식 검증 (`\frac{a}{b}`, `\alpha` 등이 온전히 보존되는지)
   * Parent-Child 청킹 검증 (부모-자식 외래키 매핑 및 메타데이터 주입)
   * BM25 인메모리 인덱싱 및 Dense+Sparse Union 로직 검증
   * BGE Reranker CPU 추론 및 0.35 컷오프 가드레일 동작 검증
2. **End-to-End 파이프라인 테스트**:
   * 수학 수식이 포함된 샘플 문서 생성 및 파싱/검색/생성 파이프라인 검증
   * 관련 없는 질문에 대한 0.35 미만 조기 종료(환각 차단) 확인

### Manual Verification
1. **Gradio UI 구동 확인**:
   * `uv run python app.py`를 실행하여 브라우저에서 Gradio UI 로드
   * PDF 업로드 -> 파싱/청킹 -> 질문 입력 -> LaTeX 수식 스트리밍 렌더링 및 디버그 인스펙터 확인
