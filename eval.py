"""Evaluation script for TheoremLens Local Math RAG pipeline.

Evaluates Retrieval Hit Rate (%) and CRAG Guardrail Cutoff accuracy using a Golden Dataset.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, List

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import config
from pipeline import MathRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalRunner")

# Golden Dataset: Math theorems, definitions, equations, and negative (out-of-scope) questions
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
    # Negative queries: Should be blocked by Guardrail (threshold = 0.35 cutoff)
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


def run_evaluation():
    print("=" * 70)
    print("🚀 TheoremLens Math RAG Golden Dataset 정량 평가 시작")
    print("=" * 70)

    pipeline = MathRAGPipeline()
    pipeline.clear_database()

    print("\n[1/3] 샘플 수학 문서 인제스천...")
    ingest_res = pipeline.ingest_markdown_text(
        SAMPLE_MATH_DOCUMENT, source_name="cauchy_schwarz_geometry.md"
    )
    print(f"✅ 인제스천 완료: 부모 청크 {ingest_res['parent_count']}개, 자식 청크 {ingest_res['child_count']}개 ({ingest_res['elapsed_seconds']:.2f}s)")

    print("\n[2/3] Golden Dataset 평가 질문 실행...")
    total_positives = sum(1 for item in GOLDEN_DATASET if item["should_pass"])
    total_negatives = sum(1 for item in GOLDEN_DATASET if not item["should_pass"])

    hit_count = 0
    guardrail_correct_blocks = 0
    results_table = []

    for item in GOLDEN_DATASET:
        q_id = item["id"]
        question = item["question"]
        should_pass = item["should_pass"]
        expected_keywords = item["expected_keywords"]

        start_t = time.time()
        search_res = pipeline.retriever.retrieve(question)
        latency = (time.time() - start_t) * 1000

        max_score = search_res.max_reranker_score
        passed = search_res.passed_guardrail

        # Check positive question hit
        if should_pass:
            if passed and search_res.top_parents:
                parent_text = " ".join(p.parent_content for p in search_res.top_parents)
                # Check if any expected keyword is in retrieved parent content
                has_hit = any(kw in parent_text for kw in expected_keywords)
                if has_hit:
                    hit_count += 1
                    status = "✅ HIT (통과 & 적중)"
                else:
                    status = "⚠️ RETRIEVED (통과했으나 키워드 불일치)"
            else:
                status = "❌ MISS (가드레일 오차단)"
        else:
            # Check negative question cutoff
            if not passed:
                guardrail_correct_blocks += 1
                status = "🛡️ BLOCKED (가드레일 정상 차단)"
            else:
                status = "❌ LEAK (환각 위험: 무관 질문 통과)"

        results_table.append({
            "id": q_id,
            "question": question[:35] + "...",
            "score": round(max_score, 4),
            "passed": passed,
            "status": status,
            "latency_ms": round(latency, 1),
        })

    print("\n[3/3] 정량 평가 결과 요약")
    print("-" * 80)
    print(f"{'ID':<6} | {'리랭커 점수':<9} | {'통과 여부':<9} | {'지연시간':<10} | {'결과 상태'}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['id']:<6} | {r['score']:<9.4f} | {str(r['passed']):<9} | {r['latency_ms']:<8.1f}ms | {r['status']}")
    print("-" * 80)

    retrieval_hit_rate = (hit_count / total_positives) * 100 if total_positives else 0
    guardrail_accuracy = (guardrail_correct_blocks / total_negatives) * 100 if total_negatives else 0

    print(f"\n📊 [핵심 지표 측정]")
    print(f"1. Retrieval Hit Rate (수학 질문 검색 적중률): {hit_count}/{total_positives} ({retrieval_hit_rate:.1f}%)")
    print(f"2. Guardrail Precision (무관 질문 환각 차단율): {guardrail_correct_blocks}/{total_negatives} ({guardrail_accuracy:.1f}%)")
    print("=" * 70)

    return {
        "hit_rate": retrieval_hit_rate,
        "guardrail_accuracy": guardrail_accuracy,
        "results": results_table,
    }


if __name__ == "__main__":
    run_evaluation()
