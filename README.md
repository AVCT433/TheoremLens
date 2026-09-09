# TheoremLens (MVP v1.1)

> **8GB VRAM 단일 로컬 머신에서 LaTeX 수식 훼손과 환각을 원천 차단하는 학술 수학 PDF 전용 고정밀 RAG 파이프라인**

---

## 🏛️ 하드웨어 분업 아키텍처

```
[사용자 수학 PDF] 
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  CPU & 시스템 RAM (VRAM 0MB 격리 연산)                 │
│  ├─ 문서 파싱: Docling (수식/표 Markdown 추출)         │
│  ├─ 청킹: Parent(1200자) - Child(250자) 계층 분할     │
│  ├─ 임베딩: nomic-embed-text (CPU 모드, 768차원)       │
│  ├─ 1차 검색: ChromaDB (Dense) + BM25 (Sparse) Union  │
│  └─ 리랭킹: BAAI/bge-reranker-base (Cross-Encoder CPU) │
│       └─ CRAG 가드레일: 최고 점수 < 0.35 시 즉시 종료  │
└────────────────────────────────────────────────────────┘
      │ (임계값 0.35 통과 시 상위 1~2개 부모 청크 주입)
      ▼
┌────────────────────────────────────────────────────────┐
│  GPU (RTX 2060 Super 8GB VRAM)                         │
│  └─ LLM 추론: qwen2.5-coder:7b (Ollama, num_ctx: 8192) │
│       ├─ 엄격한 폐쇄형 RAG (문서 외 가정/환각 원천 금지) │
│       ├─ 모든 수식 LaTeX ($...$, $$...$$) 강제        │
│       └─ 모든 진술 문장 끝 출처 [출처: ...] 태깅      │
└────────────────────────────────────────────────────────┘
      │ (MathJax 스트리밍 렌더링)
      ▼
[Gradio 웹 UI & 검색 인스펙터]
```

---

## 🚀 주요 기능 및 특징

1. **LaTeX 수식 무결성 보존**:
   - `Docling`으로 PDF 내 인라인 수식(`$...$`), 블록 수식(`$$...$$`), 다차원 표 구조를 손상 없이 마크다운으로 추출합니다.
   - Gradio 내 `latex_delimiters`를 설정하여 브라우저에서 MathJax로 실시간 렌더링합니다.

2. **Parent-Child 청킹 패턴**:
   - 검색은 작은 **자식 청크(250자)**로 정밀하게 수행하고, LLM 프롬프트에는 충분한 수학적 맥락을 담은 **부모 청크(1,200자)**를 역추적 주입합니다.

3. **하이브리드 1차 검색 (Dense + Sparse)**:
   - **Dense**: Chroma 코사인 유사도 Top-10
   - **Sparse**: LaTeX 전용 정규식 토크나이저(`r"\\[a-zA-Z]+|[a-zA-Z0-9]+|[^\s\w]"`) 기반 in-memory BM25 Top-10
   - 단순 합집합(Union, 중복 제거 후 약 14~18개) 구성 후 리랭커로 직행합니다.

4. **CRAG 스타일 0.35 임계값 가드레일 (환각 100% 차단)**:
   - `bge-reranker-base` CPU 재채점 결과 최고 점수가 `0.35` 미만이면 관련 근거 없음으로 판정하여 **LLM 호출을 즉시 건너뛰고 조기 종료**합니다.

5. **실시간 검색 인스펙터**:
   - 질문별 1차 검색 후보, BGE 리랭커 점수, 가드레일 판별 결과, 주입된 부모 청크 원문을 UI에서 투명하게 검증할 수 있습니다.

---

## 💻 빠른 시작 가이드

### 1. 사전 준비 (Prerequisites)
- **Ollama 설치 및 모델 실행**:
  ```bash
  ollama serve
  ollama pull qwen2.5-coder:7b
  ```
- **uv 가상환경**: Python 3.11+

### 2. 의존성 설치
```bash
uv sync
# 또는 uv pip install -e .
```

### 3. Gradio 웹 애플리케이션 실행
```bash
uv run python app.py
```
브라우저에서 `http://127.0.0.1:7860`으로 접속합니다.

### 4. Golden Dataset 정량 평가 실행
```bash
uv run python eval.py
```

### 5. 단위 테스트 실행
```bash
uv run python test_pipeline.py
```

---

## ⚙️ 설정값 커스터마이징 (`config.py`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `parent_chunk_size` | 1,200 | 부모 청크 글자 수 |
| `child_chunk_size` | 250 | 자식 청크 글자 수 |
| `reranker_threshold` | 0.35 | CRAG 조기 종료 임계값 |
| `llm_num_ctx` | 8192 | Ollama VRAM 컨텍스트 (필요 시 6144 또는 4096) |
| `embedding_device` | "cpu" | 임베딩 CPU 고정 (VRAM 절약) |
| `reranker_device` | "cpu" | 리랭커 CPU 고정 (VRAM 절약) |
