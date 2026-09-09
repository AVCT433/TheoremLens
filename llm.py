"""Ollama client integration with strict closed-book math RAG system prompt."""

import logging
from typing import Any, Dict, Generator, List, Optional

import ollama
from config import config
from retriever import RetrievedParent

logger = logging.getLogger(__name__)

STRICT_MATH_SYSTEM_PROMPT = """당신은 학술 수학 논문 및 문서를 정밀하게 해설하는 엄격한 폐쇄형 수학 질의응답 AI입니다.
반드시 아래의 **원칙을 100% 엄수**하여 답변하십시오.

### 엄격한 폐쇄형 답변 원칙:
1. **절대적 문서 근거 원칙 (환각 원천 차단)**:
   - 오직 아래 제공된 [참고 문서 컨텍스트]의 내용만을 근거로 답변하십시오.
   - 컨텍스트에 명시되지 않은 수학적 가정, 보조정리(Lemma), 정의, 임의의 유도 과정을 지어내거나 추측하여 추가하지 마십시오.
   - 질문에 대한 답변이 컨텍스트만으로 온전히 설명되지 않을 경우, 알 수 없거나 문서에 누락되어 있다고 정직하게 진술하십시오.

2. **LaTeX 수식 표기 표준 준수**:
   - 모든 수학적 기호, 변수, 식은 반드시 표준 LaTeX 문법으로 작성하십시오.
   - 인라인 수식은 반드시 단일 달러 기호 `$수식$`로 감싸십시오. (예: $f(x) = x^2$, $\\alpha \\in \\mathbb{R}$)
   - 독립 블록(중앙 정렬) 수식은 반드시 이중 달러 기호 `$$수식$$`로 감싸십시오. (예: $$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$)

3. **모든 진술 문장 끝 출처(Citation) 표기 의무화**:
   - 답변 내의 모든 설명이나 수학적 진술 문장의 끝에는 반드시 출처를 표기하십시오.
   - 형식: 문장 끝에 `[출처: 부모 청크 #번호]` 형태로 기재하십시오.

4. **언어**:
   - 모든 설명은 자연스러운 한국어로 명확하고 학술적으로 작성하십시오.
"""


class OllamaMathLLM:
    """Manages Ollama API calls for qwen2.5-coder:7b with VRAM guard and streaming."""

    def __init__(
        self,
        base_url: str = config.ollama_base_url,
        model: str = config.llm_model,
        num_ctx: int = config.llm_num_ctx,
        temperature: float = config.llm_temperature,
    ):
        self.base_url = base_url
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._client: Optional[ollama.Client] = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def build_prompt_context(self, parents: List[RetrievedParent]) -> str:
        """Formats the retrieved top parent chunks for context injection."""
        context_parts = []
        for idx, p in enumerate(parents, 1):
            context_parts.append(
                f"--- [부모 청크 #{idx}] (문서: {p.source}, 신뢰도 점수: {p.max_child_score:.3f}) ---\n"
                f"{p.parent_content.strip()}\n"
            )
        return "\n".join(context_parts)

    def generate_stream(
        self,
        query: str,
        parents: List[RetrievedParent],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """Streams response tokens from qwen2.5-coder:7b."""
        context_str = self.build_prompt_context(parents)

        user_content = (
            f"[참고 문서 컨텍스트]\n"
            f"{context_str}\n\n"
            f"[사용자 질문]\n"
            f"{query}\n\n"
            f"위 컨텍스트에만 근거하여 수식은 LaTeX($...$, $$...$$)로, 각 진술 끝에는 출처 [출처: 부모 청크 #n]을 포함하여 답변하십시오."
        )

        messages = [{"role": "system", "content": STRICT_MATH_SYSTEM_PROMPT}]

        # Include recent history if available
        if history:
            for turn in history[-4:]:  # keep last 2 turns
                messages.append(turn)

        messages.append({"role": "user", "content": user_content})

        options = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": 0.9,
        }

        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                # Ollama SDK 0.6+: Pydantic 모델 속성 접근 / 구버전: dict 접근 fallback
                try:
                    token = chunk.message.content or ""
                except AttributeError:
                    token = chunk.get("message", {}).get("content", "")
                yield token
        except Exception as e:
            logger.error(f"Error invoking Ollama: {e}")
            yield f"\n\n[오류 발생]: Ollama 서비스 연결 또는 모델 호출 중 문제가 발생했습니다: {str(e)}\n"
            yield f"- Ollama 실행 여부 (`ollama serve`) 및 모델 설치 (`ollama pull {self.model}`)를 확인하십시오."

    def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama is accessible and if the requested model is pulled."""
        try:
            models_info = self.client.list()
            # Ollama SDK 0.6+: Pydantic 모델 속성 접근 / 구버전: dict 접근 fallback
            try:
                models_list = models_info.models or []
                model_names = [m.model for m in models_list]
            except AttributeError:
                models_list = models_info.get("models", [])
                model_names = [m.get("model", "") for m in models_list]
            is_model_ready = any(self.model in name for name in model_names)
            return {
                "accessible": True,
                "model_ready": is_model_ready,
                "installed_models": model_names,
            }
        except Exception as e:
            return {
                "accessible": False,
                "model_ready": False,
                "error": str(e),
            }
