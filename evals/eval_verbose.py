"""Verbose evaluation script for TheoremLens Local Math RAG pipeline.

eval.py의 정량 지표에 더해, 각 질문에 대해 파이프라인 내부 모든 단계를
상세하게 출력한다:
  - Dense (임베딩) 1차 검색 히트 목록
  - Sparse (BM25) 1차 검색 히트 목록
  - Union 후보 목록 (중복 제거 후)
  - 리랭커 점수 전체 순위 (선택 / 버려진 청크 구분)
  - 가드레일(CRAG) 판정 결과
  - 최종 선택된 부모 청크 미리보기
"""

import sys
import time
from typing import Any, Dict, List

# Windows 터미널 UTF-8 출력 보장
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from theoremlens.config import config
from theoremlens.pipeline.pipeline import MathRAGPipeline
from theoremlens.pipeline.retriever import HybridSearchResult

# ──────────────────────────────────────────────────────────────
# Golden Dataset (eval.py와 동일)
# ──────────────────────────────────────────────────────────────
SAMPLE_MATH_DOCUMENT = """# Cauchy-Schwarz 부등식과 힐베르트 공간의 기하학

## 1. 내적 공간과 정의
실수 또는 복소수 체 $\\mathbb{K}$ 위의 벡터 공간 $V$에서 내적(Inner Product) $\\langle \\cdot, \\cdot \\rangle: V \\times V \\to \\mathbb{K}$는 다음 세 가지 공리를 만족하는 함수이다:
1. 켤레 대칭성(Conjugate symmetry): $\\langle x, y \\rangle = \\overline{\\langle y, x \\rangle}$
2. 첫 번째 인자에 대한 선형성: $\\langle ax + by, z \\rangle = a\\langle x, z \\rangle + b\\langle y, z \\rangle$
3. 양의 정부호성(Positive-definiteness): $\\langle x, x \\rangle \\ge 0$ 이며, $\\langle x, x \\rangle = 0 \\iff x = 0$

노름(Norm)은 내적으로부터 유도되며 다음과 같이 정의된다:
$$\\|x\\| = \\sqrt{\\langle x, x \\rangle}$$

## 2. 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality)
### 정리 2.1 (Cauchy-Schwarz Inequality)
임의의 내적 공간 $(V, \\langle \\cdot, \\cdot \\rangle)$의 원소 $x, y \\in V$에 대하여 다음 부등식이 항상 성립한다:
$$|\\langle x, y \\rangle| \\le \\|x\\| \\|y\\|$$
등호가 성립할 필요충분조건은 두 벡터 $x$와 $y$가 선형 종속(Linearly Dependent), 즉 어떤 스칼라 $\\lambda \\in \\mathbb{K}$에 대해 $x = \\lambda y$ 또는 $y = 0$인 경우이다.

### 증명 개요
$y = 0$이면 양변이 $0$이 되어 자명하다. $y \\neq 0$이라 가정하고, 임의의 스칼라 $t \\in \\mathbb{R}$에 대해 벡터 $z = x - t \\frac{\\langle x, y \\rangle}{\\|y\\|^2} y$를 고려한다.
양의 정부호성에 의해:
$$\\|z\\|^2 = \\langle z, z \\rangle \\ge 0$$
이를 전개하면 판별식이 $0$ 이하가 되어 부등식 $|\\langle x, y \\rangle|^2 \\le \\|x\\|^2 \\|y\\|^2$가 유도된다.

## 3. 삼각 부등식 (Minkowski Inequality)
코시-슈바르츠 부등식의 직접적인 귀결로서, 임의의 $x, y \\in V$에 대해 노름의 삼각 부등식이 성립한다:
$$\\|x + y\\| \\le \\|x\\| + \\|y\\|$$
이로써 모든 내적 공간은 유도된 노름에 의해 노름 공간이 되며, 이 노름이 완비성(Completeness)을 만족할 때 이를 힐베르트 공간(Hilbert Space)이라 부른다.
"""

GOLDEN_DATASET = [
    {
        "id": "q1",
        "question": "내적의 세 가지 공리에는 무엇이 있는가?",
        "expected_keywords": ["켤레 대칭성", "선형성", "양의 정부호성"],
        "should_pass": True,
    },
    {
        "id": "q2",
        "question": "정리 2.1 코시-슈바르츠 부등식의 수식 표현은 어떻게 되는가?",
        "expected_keywords": ["|\\langle x, y \\rangle|", "\\|x\\|", "\\|y\\|"],
        "should_pass": True,
    },
    {
        "id": "q3",
        "question": "코시-슈바르츠 부등식에서 등호가 성립할 필요충분조건은 무엇인가?",
        "expected_keywords": ["선형 종속", "Linearly Dependent"],
        "should_pass": True,
    },
    {
        "id": "q4",
        "question": "내적 공간에서 노름(Norm)은 어떻게 유도 정의되는가?",
        "expected_keywords": ["\\sqrt{\\langle x, x \\rangle}", "\\|x\\|"],
        "should_pass": True,
    },
    {
        "id": "q5",
        "question": "힐베르트 공간(Hilbert Space)의 정의는 무엇인가?",
        "expected_keywords": ["완비성", "Completeness", "노름 공간"],
        "should_pass": True,
    },
    # 음성 질문: 가드레일이 차단해야 함 (threshold = 0.35)
    {
        "id": "neg1",
        "question": "피자의 도우를 바삭하게 굽는 온도와 발효 시간은 얼마인가요?",
        "expected_keywords": [],
        "should_pass": False,
    },
    {
        "id": "neg2",
        "question": "리액트 컴포넌트 생명주기와 useEffect의 의존성 배열 동작 원리는?",
        "expected_keywords": [],
        "should_pass": False,
    },
]


# ──────────────────────────────────────────────────────────────
# 헬퍼: 청크 내용 미리보기 (한 줄, 최대 width 글자)
# ──────────────────────────────────────────────────────────────
# def _preview(text: str, width: int = 90) -> str:
#     """텍스트를 한 줄로 압축해서 미리보기."""
#     one_line = " ".join(text.split())
#     return one_line[:width] + ("…" if len(one_line) > width else "")
def _preview(text: str, width: int = 90) -> str:
    """텍스트를 한 줄로 압축해서 앞부분과 끝부분을 함께 미리보기."""
    one_line = " ".join(text.split())
    if len(one_line) <= width:
        return one_line
    ellipsis = "..."
    remaining = width - len(ellipsis)
    if remaining <= 0:
        return one_line[:width]
    front_len = (remaining + 1) // 2
    back_len = remaining // 2
    return f"{one_line[:front_len]}{ellipsis}{one_line[-back_len:]}"


def _sep(char: str = "─", width: int = 80) -> str:
    return char * width


# ──────────────────────────────────────────────────────────────
# 상세 출력: 단계별 검색 결과 분석
# ──────────────────────────────────────────────────────────────
def print_verbose_result(
    q_id: str,
    question: str,
    should_pass: bool,
    expected_keywords: List[str],
    result: HybridSearchResult,
    latency_ms: float,
) -> Dict[str, Any]:
    """한 질문에 대한 파이프라인 전 단계를 상세 출력하고 평가 결과를 반환한다."""

    threshold = config.reranker_threshold

    print()
    print(_sep("═"))
    type_label = "✅ 양성 질문 (should_pass)" if should_pass else "🚫 음성 질문 (should_block)"
    print(f"  [{q_id}] {type_label}")
    print(f"  질문: {question}")
    print(_sep("═"))

    # ── STEP 1. Dense 1차 검색 히트 ──────────────────────────
    print(f"\n  [STEP 1] Dense 검색 결과 (임베딩 유사도, Top-{config.dense_top_k})")
    print(_sep())
    if result.dense_hits:
        for i, h in enumerate(result.dense_hits, 1):
            score_str = f"{h.get('score', 0):.4f}" if "score" in h else "N/A"
            print(f"  {i:>2}. [child_id: {h['child_id'][:24]}]  점수: {score_str}")
            print(f"      └ {_preview(h['content'])}")
    else:
        print("  (결과 없음)")

    # ── STEP 2. Sparse BM25 1차 검색 히트 ────────────────────
    print(f"\n  [STEP 2] Sparse 검색 결과 (BM25, Top-{config.sparse_top_k})")
    print(_sep())
    if result.sparse_hits:
        for i, h in enumerate(result.sparse_hits, 1):
            print(f"  {i:>2}. [child_id: {h['child_id'][:24]}]  BM25 점수: {h.get('score', 0):.4f}")
            print(f"      └ {_preview(h['content'])}")
    else:
        print("  (결과 없음)")

    # ── STEP 3. Union 후보 (중복 제거) ───────────────────────
    union = result.union_candidates
    dense_ids = {h["child_id"] for h in result.dense_hits}
    sparse_ids = {h["child_id"] for h in result.sparse_hits}

    print(f"\n  [STEP 3] Union 후보 ({len(union)}개, Dense ∪ Sparse 중복 제거)")
    print(_sep())
    if union:
        for i, c in enumerate(union, 1):
            cid = c["child_id"]
            origin_tags = []
            if cid in dense_ids:
                origin_tags.append("Dense")
            if cid in sparse_ids:
                origin_tags.append("Sparse")
            tag_str = "+".join(origin_tags) if origin_tags else "?"
            print(f"  {i:>2}. [{tag_str:>13}] [child_id: {cid[:24]}]")
            print(f"      └ {_preview(c['content'])}")
    else:
        print("  (후보 없음)")

    # ── STEP 4. 리랭커 전체 순위 ──────────────────────────────
    reranked = result.reranked_candidates
    print(f"\n  [STEP 4] 리랭커 점수 전체 순위 ({len(reranked)}개 후보, 임계값={threshold})")
    print(_sep())

    # 최종 선택된 parent_id 집합
    selected_parent_ids = {p.parent_id for p in result.top_parents}

    if reranked:
        for i, c in enumerate(reranked, 1):
            score = c.get("reranker_score", 0.0)
            pid = c.get("parent_id", "")
            cid = c["child_id"]

            if score >= threshold:
                if pid in selected_parent_ids:
                    verdict = "🟢 최종 선택됨"
                else:
                    verdict = "🟡 임계값 이상이지만 부모 중복으로 버려짐"
            else:
                verdict = f"🔴 버려짐 (score {score:.4f} < threshold {threshold})"

            print(f"  {i:>2}. score={score:.4f}  {verdict}")
            print(f"      child_id  : {cid[:36]}")
            print(f"      parent_id : {pid[:36]}")
            print(f"      내용      : {_preview(c['content'], 80)}")
    else:
        print("  (리랭킹 후보 없음)")

    # ── STEP 5. 가드레일 판정 ─────────────────────────────────
    print(f"\n  [STEP 5] 가드레일(CRAG) 판정")
    print(_sep())
    max_score = result.max_reranker_score
    passed = result.passed_guardrail
    print(f"  최고 리랭커 점수 : {max_score:.4f}  (임계값: {threshold})")
    if passed:
        print(f"  판정: ✅ 통과 → LLM 응답 허용")
    else:
        msg = result.guardrail_message or ""
        print(f"  판정: 🛡️  차단 → LLM 호출 없음")
        if msg:
            print(f"  메시지: {msg}")

    # ── STEP 6. 최종 선택된 부모 청크 ────────────────────────
    print(f"\n  [STEP 6] 최종 선택된 부모 청크 ({len(result.top_parents)}개)")
    print(_sep())
    if result.top_parents:
        for i, p in enumerate(result.top_parents, 1):
            print(f"  [{i}] parent_id  : {p.parent_id[:36]}")
            print(f"       source     : {p.source}")
            print(f"       max score  : {p.max_child_score:.4f}")
            print(f"       매칭 자식  : {len(p.matched_children)}개")
            print(f"       내용 미리보기:")
            lines = p.parent_content.strip().splitlines()
            for ln in lines[:6]:
                print(f"         {ln}")
            if len(lines) > 6:
                print(f"         … (총 {len(lines)}줄)")
    else:
        print("  (선택된 부모 청크 없음 — 가드레일에 의해 차단됨)")

    # ── STEP 7. 이 질문의 최종 평가 판정 ─────────────────────
    print(f"\n  [STEP 7] 최종 평가 판정")
    print(_sep())

    is_hit = False
    guardrail_ok = None

    if should_pass:
        if passed and result.top_parents:
            parent_text = " ".join(p.parent_content for p in result.top_parents)
            has_hit = any(kw in parent_text for kw in expected_keywords)
            if has_hit:
                final_status = "✅ HIT  (통과 & 키워드 적중)"
                is_hit = True
            else:
                missing = [kw for kw in expected_keywords if kw not in parent_text]
                final_status = f"⚠️  RETRIEVED  (통과했으나 키워드 불일치)"
                print(f"  미적중 키워드: {missing}")
        else:
            final_status = "❌ MISS  (가드레일 오차단)"
    else:
        if not passed:
            final_status = "🛡️  BLOCKED  (가드레일 정상 차단)"
            guardrail_ok = True
        else:
            final_status = "❌ LEAK  (환각 위험: 무관 질문 통과)"
            guardrail_ok = False

    print(f"  결과      : {final_status}")
    print(f"  지연 시간 : {latency_ms:.1f} ms")

    return {
        "id": q_id,
        "question": question[:40] + ("…" if len(question) > 40 else ""),
        "score": round(max_score, 4),
        "passed": passed,
        "status": final_status,
        "latency_ms": round(latency_ms, 1),
        "is_hit": is_hit,
        "guardrail_ok": guardrail_ok,
        "should_pass": should_pass,
    }


# ──────────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────────
def run_verbose_evaluation():
    print(_sep("═", 80))
    print("🔬 TheoremLens Math RAG — 상세 파이프라인 진단 평가 (eval_verbose)")
    print(_sep("═", 80))

    pipeline = MathRAGPipeline()
    pipeline.clear_database()

    print("\n[1/3] 샘플 수학 문서 인제스천...")
    ingest_res = pipeline.ingest_markdown_text(
        SAMPLE_MATH_DOCUMENT, source_name="cauchy_schwarz_geometry.md"
    )
    print(
        f"✅ 인제스천 완료: 부모 청크 {ingest_res['parent_count']}개, "
        f"자식 청크 {ingest_res['child_count']}개 "
        f"({ingest_res['elapsed_seconds']:.2f}s)"
    )

    print("\n[2/3] Golden Dataset 상세 평가 시작...")

    total_positives = sum(1 for item in GOLDEN_DATASET if item["should_pass"])
    total_negatives = sum(1 for item in GOLDEN_DATASET if not item["should_pass"])

    hit_count = 0
    guardrail_correct_blocks = 0
    all_results = []

    for item in GOLDEN_DATASET:
        start_t = time.time()
        search_res = pipeline.retriever.retrieve(item["question"])
        latency = (time.time() - start_t) * 1000

        row = print_verbose_result(
            q_id=item["id"],
            question=item["question"],
            should_pass=item["should_pass"],
            expected_keywords=item["expected_keywords"],
            result=search_res,
            latency_ms=latency,
        )
        all_results.append(row)

        if row["should_pass"] and row["is_hit"]:
            hit_count += 1
        if not row["should_pass"] and row["guardrail_ok"]:
            guardrail_correct_blocks += 1

    # ── 최종 요약 ────────────────────────────────────────────
    print()
    print(_sep("═", 80))
    print("[3/3] 📊 최종 정량 평가 요약")
    print(_sep("═", 80))

    print(f"\n{'ID':<6} | {'리랭커 점수':>9} | {'통과':>5} | {'지연(ms)':>8} | 결과")
    print(_sep("-", 80))
    for r in all_results:
        print(
            f"{r['id']:<6} | {r['score']:>9.4f} | {str(r['passed']):>5} | "
            f"{r['latency_ms']:>7.1f}  | {r['status']}"
        )
    print(_sep("-", 80))

    retrieval_hit_rate = (hit_count / total_positives * 100) if total_positives else 0.0
    guardrail_accuracy = (guardrail_correct_blocks / total_negatives * 100) if total_negatives else 0.0

    print(
        f"\n  1. Retrieval Hit Rate  (양성 키워드 적중률)   : "
        f"{hit_count}/{total_positives}  ({retrieval_hit_rate:.1f}%)"
    )
    print(
        f"  2. Guardrail Precision (무관 질문 환각 차단율): "
        f"{guardrail_correct_blocks}/{total_negatives}  ({guardrail_accuracy:.1f}%)"
    )
    print(_sep("═", 80))

    return {
        "hit_rate": retrieval_hit_rate,
        "guardrail_accuracy": guardrail_accuracy,
        "results": all_results,
    }


if __name__ == "__main__":
    run_verbose_evaluation()
