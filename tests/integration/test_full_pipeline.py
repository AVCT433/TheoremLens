"""
TheoremLens 전체 파이프라인 E2E 테스트
========================================
흐름:
  1. [환경 체크]  Ollama 헬스 체크 & 모델 설치 확인
  2. [수집]       샘플 수학 마크다운 텍스트 ingest_markdown_text
  3. [저장]       ChromaDB 청크 삽입 & BM25 인덱스 빌드
  4. [검색]       Dense(Chroma) + Sparse(BM25) 하이브리드 + Cross-Encoder 재랭킹 + CRAG 가드레일
  5. [LLM 생성]   Ollama(qwen2.5-coder:7b) 스트리밍 응답 수신
  6. [가드레일]   무관한 쿼리로 CRAG 조기종료 동작 검증
  7. [정리]       테스트용 DB 컬렉션 초기화

사용:
  uv run python test_full_pipeline.py
"""

import logging
import sys
import time
import io

# Windows 터미널 인코딩 문제 방지 (cp949 -> utf-8 강제)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 로깅 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("E2E-Test")


# ── 색상 출력 헬퍼 ───────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"


def section(title: str) -> None:
    bar = "-" * 60
    print(f"\n{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")


def ok(msg: str)   -> None: print(f"  {C.GREEN}[OK]  {msg}{C.RESET}")
def warn(msg: str) -> None: print(f"  {C.YELLOW}[!!]  {msg}{C.RESET}")
def fail(msg: str) -> None: print(f"  {C.RED}[NG]  {msg}{C.RESET}")
def info(msg: str) -> None: print(f"  {C.BLUE}[>>]  {msg}{C.RESET}")


# ── 샘플 수학 문서 ───────────────────────────────────────────────────────────
SAMPLE_MATH_DOC = r"""
## 정리 1: 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality)

내적 공간 $(V, \langle\cdot,\cdot\rangle)$ 에서 임의의 벡터 $u, v \in V$ 에 대해 다음이 성립한다.

$$|\langle u, v \rangle|^2 \leq \langle u, u \rangle \cdot \langle v, v \rangle$$

즉, $|\langle u, v \rangle| \leq \|u\| \cdot \|v\|$ 이며 등호는 $u, v$ 가 선형 종속일 때만 성립한다.

### 증명

$v = 0$ 이면 자명하다. $v \neq 0$ 으로 가정한다.
임의의 $t \in \mathbb{R}$ 에 대해

$$0 \leq \langle u - tv, u - tv \rangle = \langle u,u \rangle - 2t\langle u,v \rangle + t^2\langle v,v \rangle$$

이 이차함수가 항상 음이 아니려면 판별식이 0 이하이어야 한다:

$$\Delta = 4\langle u,v \rangle^2 - 4\langle u,u \rangle\langle v,v \rangle \leq 0$$

따라서 $\langle u,v \rangle^2 \leq \langle u,u \rangle\langle v,v \rangle$ 가 성립한다. $\blacksquare$

---

## 정리 2: 바나흐 고정점 정리 (Banach Fixed-Point Theorem)

완비 거리 공간 $(X, d)$ 에서 $T: X \to X$ 가 어떤 $0 \leq k < 1$ 에 대해

$$d(T(x), T(y)) \leq k \cdot d(x, y)$$

를 만족하면, $T$ 는 유일한 고정점 $x^* \in X$ ($T(x^*) = x^*$)를 가진다.

### 증명 스케치

$x_0 \in X$ 를 선택하고 $x_{n+1} = T(x_n)$ 으로 정의하면 귀납적으로

$$d(x_{n+1}, x_n) \leq k^n \cdot d(x_1, x_0)$$

이므로 코시 수열이다. $X$ 가 완비이므로 $x^* = \lim_{n\to\infty} x_n$ 이 존재하며
$T$ 의 연속성에 의해 $T(x^*) = x^*$ 이다. $\blacksquare$

---

## 정리 3: 스펙트럼 정리 (Spectral Theorem for Symmetric Matrices)

실수 대칭 행렬 $A \in \mathbb{R}^{n \times n}$ ($A = A^T$) 은 항상 실수 고유값만을 가지며
다음과 같이 직교 분해된다:

$$A = Q \Lambda Q^T$$

여기서 $Q$ 는 직교 행렬 ($Q^T Q = I$), $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ 이다.
"""

TEST_QUERY = "코시-슈바르츠 부등식의 증명 과정에서 판별식을 이용하는 방법을 설명해줘."
IRRELEVANT_QUERY = "오늘 서울 날씨는 어때?"


# ── 메인 테스트 함수 ─────────────────────────────────────────────────────────
def run_e2e_test() -> None:
    total_start = time.time()

    # STEP 0: 모듈 임포트
    section("STEP 0 | 모듈 임포트 확인")
    try:
        from src.pipeline.pipeline import MathRAGPipeline  # noqa: F401
        from src.llm import OllamaMathLLM
        ok("모든 TheoremLens 모듈 임포트 성공")
    except ImportError as e:
        fail(f"임포트 실패: {e}")
        sys.exit(1)

    # STEP 1: Ollama 헬스 체크
    section("STEP 1 | Ollama 헬스 체크")
    llm_probe = OllamaMathLLM()
    health = llm_probe.check_health()
    if health["accessible"]:
        ok(f"Ollama 서버 접근 가능  (URL: {llm_probe.base_url})")
        info(f"설치된 모델: {health.get('installed_models', [])}")
        if health["model_ready"]:
            ok(f"대상 모델 준비 완료: {llm_probe.model}")
        else:
            warn(
                f"모델 '{llm_probe.model}' 이 설치되지 않았습니다. "
                f"'ollama pull {llm_probe.model}' 을 실행하세요."
            )
    else:
        warn(f"Ollama 서버 접근 불가: {health.get('error')}")
        warn("LLM 단계는 오류 메시지를 반환할 수 있습니다.")

    # STEP 2: 파이프라인 초기화
    section("STEP 2 | MathRAGPipeline 초기화")
    from src.pipeline.pipeline import MathRAGPipeline

    t0 = time.time()
    pipeline = MathRAGPipeline()
    ok(f"파이프라인 초기화 완료  ({time.time() - t0:.2f}s)")

    # STEP 3: DB 클린 슬레이트
    section("STEP 3 | 테스트 전 DB 초기화 (클린 슬레이트)")
    pipeline.clear_database()
    ok("ChromaDB 컬렉션 & BM25 인덱스 초기화 완료")

    # STEP 4: 마크다운 문서 수집
    section("STEP 4 | 샘플 수학 문서 수집  (ingest_markdown_text)")
    info(f"문서 크기: {len(SAMPLE_MATH_DOC)} 자")
    t0 = time.time()
    result = pipeline.ingest_markdown_text(
        SAMPLE_MATH_DOC,
        source_name="sample_math_theorems.md",
    )
    elapsed_ingest = time.time() - t0
    ok(f"수집 완료  ({elapsed_ingest:.2f}s)")
    info(f"소스          : {result['source']}")
    info(f"부모 청크 수  : {result['parent_count']}")
    info(f"자식 청크 수  : {result['child_count']}")
    info(f"마크다운 미리보기 (첫 200자):\n    {result['markdown_preview'][:200]}")

    assert result["parent_count"] > 0, "부모 청크가 1개 이상이어야 합니다."
    assert result["child_count"] > 0, "자식 청크가 1개 이상이어야 합니다."

    # STEP 5: 하이브리드 검색 & 재랭킹 & 가드레일
    section("STEP 5 | 하이브리드 검색 + Cross-Encoder 재랭킹 + CRAG 가드레일")
    info(f"쿼리: {TEST_QUERY!r}")
    t0 = time.time()
    search_result = pipeline.retriever.retrieve(TEST_QUERY)
    elapsed_retrieve = time.time() - t0

    ok(f"검색 완료  ({elapsed_retrieve:.2f}s)")
    info(f"Dense  히트 수      : {len(search_result.dense_hits)}")
    info(f"Sparse 히트 수      : {len(search_result.sparse_hits)}")
    info(f"Union  후보 수      : {len(search_result.union_candidates)}")
    info(f"재랭킹 후보 수      : {len(search_result.reranked_candidates)}")
    info(f"최고 재랭킹 점수    : {search_result.max_reranker_score:.4f}")
    info(f"CRAG 가드레일 통과  : {'YES' if search_result.passed_guardrail else 'NO'}")

    if search_result.passed_guardrail:
        ok(f"가드레일 통과 -> 선택된 부모 청크: {len(search_result.top_parents)}개")
        for i, p in enumerate(search_result.top_parents, 1):
            info(
                f"  부모 #{i} | 점수: {p.max_child_score:.4f} | 소스: {p.source}\n"
                f"    미리보기: {p.parent_content[:120].replace(chr(10), ' ')}..."
            )
    else:
        warn(f"가드레일 차단: {search_result.guardrail_message}")

    # STEP 6: LLM 스트리밍 응답
    section("STEP 6 | LLM 스트리밍 응답  (Ollama)")
    info(f"모델  : {pipeline.llm.model}")
    info(f"쿼리  : {TEST_QUERY!r}")

    bar = "-" * 60
    print(f"\n{bar}\n  LLM 응답 (스트리밍 출력):\n{bar}")

    t0 = time.time()
    llm_stream, _ = pipeline.query(TEST_QUERY)

    full_response = ""
    for token in llm_stream:
        print(token, end="", flush=True)
        full_response += token
    elapsed_llm = time.time() - t0

    print(f"\n{bar}")
    ok(f"LLM 응답 수신 완료  ({elapsed_llm:.2f}s, {len(full_response)}자)")
    assert len(full_response) > 0, "LLM 응답이 비어 있습니다."

    # STEP 7: 가드레일 동작 검증 (무관한 쿼리)
    section("STEP 7 | CRAG 가드레일 검증  (무관한 쿼리)")
    info(f"쿼리: {IRRELEVANT_QUERY!r}")

    irr_stream, irr_result = pipeline.query(IRRELEVANT_QUERY)
    _ = "".join(irr_stream)

    if not irr_result.passed_guardrail:
        ok(
            f"가드레일 정상 작동 -> 조기 종료 "
            f"(최고 점수: {irr_result.max_reranker_score:.4f})"
        )
        info(f"메시지: {irr_result.guardrail_message}")
    else:
        warn(
            f"가드레일이 예상과 달리 통과됨 "
            f"(점수: {irr_result.max_reranker_score:.4f}). "
            "임계값 조정을 고려하세요."
        )

    # STEP 8: DB 정리
    section("STEP 8 | 테스트 후 DB 초기화")
    pipeline.clear_database()
    ok("ChromaDB 컬렉션 초기화 완료  (테스트 데이터 삭제)")

    # 최종 요약
    total_elapsed = time.time() - total_start
    section("[DONE] 전체 파이프라인 E2E 테스트 완료")
    info(f"수집 시간    : {elapsed_ingest:.2f}s")
    info(f"검색 시간    : {elapsed_retrieve:.2f}s")
    info(f"LLM 응답     : {elapsed_llm:.2f}s")
    info(f"총 소요 시간 : {total_elapsed:.2f}s")
    print(f"\n  {C.BOLD}{C.GREEN}[SUCCESS] 모든 단계 정상 완료!{C.RESET}\n")


if __name__ == "__main__":
    run_e2e_test()
