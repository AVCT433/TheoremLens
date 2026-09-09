"""TheoremLens - High-Precision Local Math RAG Web Interface (Gradio)."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from config import config
from pipeline import MathRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TheoremLensApp")

# Initialize Pipeline singleton
pipeline = MathRAGPipeline()

# MathJax LaTeX delimiter configuration for Gradio Chatbot
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
}
.math-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f8fafc;
    padding: 1.25rem 1.75rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    border: 1px solid #334155;
}
.math-banner h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.math-banner p {
    margin: 0.35rem 0 0 0;
    font-size: 0.92rem;
    color: #94a3b8;
}
.badge-pass {
    display: inline-block;
    background-color: #065f46;
    color: #34d399;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8rem;
}
.badge-fail {
    display: inline-block;
    background-color: #991b1b;
    color: #f87171;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8rem;
}
"""


def process_pdf_upload(file_obj) -> Tuple[str, str]:
    """Handles PDF file upload, moves to uploads folder, and runs ingestion."""
    if file_obj is None:
        return "⚠️ 업로드할 PDF 파일을 선택해주세요.", ""

    try:
        temp_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
        target_path = config.upload_directory / temp_path.name
        shutil.copy(temp_path, target_path)

        result = pipeline.ingest_pdf(target_path)
        status_msg = (
            f"✅ **문서 색인 완료!**\n\n"
            f"- **문서명**: `{result['source']}`\n"
            f"- **부모 청크(1,200자)**: {result['parent_count']}개\n"
            f"- **자식 청크(250자)**: {result['child_count']}개\n"
            f"- **BM25 & Chroma 색인 소요**: {result['elapsed_seconds']:.2f}초"
        )
        return status_msg, result["markdown_preview"]
    except Exception as e:
        logger.exception("Error processing PDF upload")
        return f"❌ **문서 색인 실패**: {str(e)}", ""


def reset_database() -> str:
    """Clears Chroma collection and BM25 index."""
    try:
        pipeline.clear_database()
        return "🧹 데이터베이스가 초기화되었습니다."
    except Exception as e:
        return f"❌ 초기화 실패: {str(e)}"


def check_ollama_status() -> str:
    """Checks the local Ollama daemon and model status."""
    health = pipeline.llm.check_health()
    if not health.get("accessible"):
        return f"🔴 Ollama 미연결 (오류: {health.get('error', '서비스 미구동')})"
    if not health.get("model_ready"):
        installed = ", ".join(health.get("installed_models", []))
        return f"🟡 Ollama 연결됨 | 모델 `{config.llm_model}` 없음 (설치됨: {installed or '없음'})"
    return f"🟢 Ollama 준비 완료 (`{config.llm_model}` 준비됨)"


def respond(
    message: str,
    chat_history: List[Dict[str, str]],
    threshold: float,
    num_ctx: int,
    top_parents: int,
):
    """Handles user query with streaming response and inspector panel updates."""
    if not message.strip():
        yield chat_history, "", "질문을 입력해주세요."
        return

    # Update dynamic runtime settings
    pipeline.retriever.threshold = float(threshold)
    pipeline.retriever.reranker.threshold = float(threshold)
    pipeline.retriever.final_top_parents = int(top_parents)
    pipeline.llm.num_ctx = int(num_ctx)

    # Convert chat_history to messages format if needed
    history_messages = []
    for turn in chat_history:
        history_messages.append(turn)

    # Append user turn
    new_history = list(chat_history)
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": ""})

    stream_gen, search_result = pipeline.query(message, history=history_messages)

    # Format inspector information
    inspector_text = format_inspector_info(search_result, threshold)

    # Stream assistant response
    accumulated_text = ""
    for token in stream_gen:
        accumulated_text += token
        new_history[-1]["content"] = accumulated_text
        yield new_history, "", inspector_text


def format_inspector_info(result, threshold: float) -> str:
    """Formats the debug inspector Markdown content."""
    guardrail_status = (
        f'<span class="badge-pass">PASS (통과)</span>'
        if result.passed_guardrail
        else f'<span class="badge-fail">CUTOFF (차단: 환각 방지)</span>'
    )

    text = [
        f"### 🔍 검색 & 가드레일 실시간 진단",
        f"- **가드레일 상태**: {guardrail_status}",
        f"- **최고 리랭킹 점수 (BGE Cross-Encoder)**: `{result.max_reranker_score:.4f}` (임계값: `{threshold:.2f}`)",
        f"- **1차 검색 후보**: Dense `{len(result.dense_hits)}`개 + BM25 `{len(result.sparse_hits)}`개 → 중복 제거 합집합 `{len(result.union_candidates)}`개",
        "",
        "---",
        "#### 📑 주입된 부모 컨텍스트 (상위 1~2개)",
    ]

    if result.top_parents:
        for idx, parent in enumerate(result.top_parents, 1):
            text.append(
                f"**[부모 청크 #{idx}]** (최고 자식 점수: `{parent.max_child_score:.4f}`, 출처: `{parent.source}`)\n"
                f"```text\n{parent.parent_content.strip()}\n```\n"
            )
    else:
        text.append("*가드레일에 의해 부모 컨텍스트 주입이 차단되었습니다.*")

    text.append("\n---\n#### 📊 상위 리랭킹 후보 자식 청크 (Top 5)")
    for idx, cand in enumerate(result.reranked_candidates[:5], 1):
        score = cand.get("reranker_score", 0.0)
        ret_type = cand.get("retrieval_type", "unknown")
        text.append(f"{idx}. **[{ret_type.upper()}] 점수: `{score:.4f}`** | ID: `{cand.get('child_id')}`")
        snippet = cand.get("content", "").replace("\n", " ")[:90]
        text.append(f"   > {snippet}...")

    return "\n".join(text)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TheoremLens - Local Math RAG", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="math-banner">
                <h1>📐 TheoremLens Local Math RAG (MVP v1.1)</h1>
                <p>8GB VRAM 단일 로컬 환경에서 수식 무결성($...$, $$...$$)과 환각 원천 차단(CRAG 가드레일)을 실현하는 학술 수학 전용 RAG</p>
            </div>
            """
        )

        with gr.Row():
            # Left Sidebar: Document Management & System Config
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 📄 수학 PDF 문서 색인")
                pdf_input = gr.File(
                    label="수학 논문 / 교재 PDF 업로드",
                    file_types=[".pdf"],
                    file_count="single",
                )
                ingest_btn = gr.Button("🚀 문서 분석 및 색인 시작 (Docling)", variant="primary")
                ingest_status = gr.Markdown("문서를 업로드한 후 색인 버튼을 눌러주세요.")

                with gr.Accordion("📝 파싱된 원문 미리보기 (Markdown)", open=False):
                    markdown_preview = gr.TextArea(
                        label="Docling Markdown 추출 결과",
                        lines=8,
                        interactive=False,
                    )

                gr.Markdown("---")
                gr.Markdown("### ⚙️ 하이퍼파라미터 제어")
                threshold_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.70,
                    value=config.reranker_threshold,
                    step=0.01,
                    label="CRAG 리랭커 차단 임계값 (Threshold)",
                    info="0.35 미만 점수는 환각 방지를 위해 LLM 호출을 즉시 생략합니다.",
                )
                num_ctx_dropdown = gr.Dropdown(
                    choices=[4096, 6144, 8192],
                    value=config.llm_num_ctx,
                    label="Ollama VRAM 컨텍스트 창 (num_ctx)",
                    info="RTX 2060S 8GB VRAM 스왑 발생 시 6144 또는 4096으로 하향 조정",
                )
                top_parents_slider = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=config.final_top_parents,
                    step=1,
                    label="최종 주입 부모 청크 수",
                )

                gr.Markdown("---")
                gr.Markdown("### 🖥️ 시스템 & DB 상태")
                ollama_status_box = gr.Markdown(value=check_ollama_status())
                refresh_status_btn = gr.Button("🔄 Ollama 상태 새로고침", size="sm")
                reset_db_btn = gr.Button("🧹 Chroma DB 초기화", size="sm", variant="stop")

            # Right Main Area: Chatbot & Inspection Panel
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="TheoremLens 수학 질의응답",
                    height=520,
                    latex_delimiters=LATEX_DELIMITERS,
                    type="messages",
                    placeholder="수학 논문 또는 수식에 대해 질문하십시오. (예: '정리 2.1의 가정과 증명 개요는 무엇인가?')",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="수학적 질문을 입력하세요... (Enter 또는 전송 버튼)",
                        show_label=False,
                        scale=9,
                        autofocus=True,
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)

                clear_chat_btn = gr.Button("💬 대화창 비우기", size="sm")

                with gr.Accordion("🔬 검색 및 가드레일 진단 인스펙터 (Debug Inspector)", open=True):
                    inspector_output = gr.Markdown(
                        value="*질문을 입력하면 Dense/BM25 검색 후보, BGE 리랭킹 점수, 주입된 부모 청크 원문이 여기에 표시됩니다.*"
                    )

        # Event Handlers
        ingest_btn.click(
            fn=process_pdf_upload,
            inputs=[pdf_input],
            outputs=[ingest_status, markdown_preview],
        )

        refresh_status_btn.click(
            fn=check_ollama_status,
            inputs=[],
            outputs=[ollama_status_box],
        )

        reset_db_btn.click(
            fn=reset_database,
            inputs=[],
            outputs=[ingest_status],
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, threshold_slider, num_ctx_dropdown, top_parents_slider],
            outputs=[chatbot, msg_input, inspector_output],
        )

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, threshold_slider, num_ctx_dropdown, top_parents_slider],
            outputs=[chatbot, msg_input, inspector_output],
        )

        clear_chat_btn.click(
            fn=lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, msg_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=False,
    )
