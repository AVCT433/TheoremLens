import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Union, Any, Optional

from transformers import AutoTokenizer

from theoremlens.config import config


# Content 노드가 가장 깊은 부모(최하위 소제목)를 찾을 수 있도록 사용하는 가상 최대 레벨
MAX_LEVEL = 9999

# =============================================================================
# 토크나이저 Lazy 초기화 (모듈 레벨 싱글턴)
# =============================================================================
_tokenizer = None


def _get_tokenizer():
    """토크나이저를 최초 1회만 로드하고 이후에는 캐시된 인스턴스를 반환합니다."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    return _tokenizer


def count_tokens(text: str) -> int:
    """주어진 텍스트의 토큰 수를 반환합니다."""
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


# =============================================================================
# 구조 기반(Structure-First) Parent-Child 시맨틱 청커
# =============================================================================
def chunk_mineru_math_doc(content_list: list) -> list[dict]:
    """
    MinerU가 출력한 JSON 블록 리스트를 단일 패스로 순회하며,
    임베딩과 JSON 직렬화에 최적화된 1차원 평면 리스트를 반환합니다.

    Args:
        content_list: MinerU 파싱 결과 블록 리스트
            [{"type": "text", "text_level": 1, "text": "# Chapter 1"}, ...]

    Returns:
        list[dict]: 청크 딕셔너리의 평면 리스트 (스키마는 모듈 docstring 참조)
    """
    # 가상 목차 정규식 (라벨과 나머지 본문 분리용)
    virtual_header_re = re.compile(r"^(Theorem|Lemma|Proposition|Corollary|Definition)(.*)", re.IGNORECASE | re.DOTALL)

    # 독립 엔티티 정규식 (라벨과 나머지 본문 분리용)
    entity_patterns = [
        ("proof", re.compile(r"^(Proof)(.*)", re.IGNORECASE | re.DOTALL)),
        ("example", re.compile(r"^(Examples?)(.*)", re.IGNORECASE | re.DOTALL)),
        ("remark", re.compile(r"^(Remark|Note|Solution)(.*)", re.IGNORECASE | re.DOTALL)),
    ]

    # ── 결과 리스트 & 상태 변수 ──
    result: list[dict] = []
    global_index: int = 0

    # 계층 트리 추적: {레벨(int): chunk_id(str)}
    active_parents: dict[int, str] = {}

    # 콘텐츠 누적 버퍼
    current_content_blocks: list[str] = []
    current_tokens: int = 0
    current_page_indices: list[int] = []
    
    # 상태 머신
    current_state_type = "content"
    last_block_type = None  # "text" | "equation" | None

    # ─────────────────────────────────────────────────────────────
    #  헬퍼: active_parents에서 parent_id 결정
    # ─────────────────────────────────────────────────────────────
    def _find_parent_id(current_level: int) -> Optional[str]:
        """
        현재 레벨보다 작은(상위) 레벨 중 최댓값의 chunk_id를 반환합니다.
        Content 노드는 current_level=∞ 로 호출하여 가장 깊은 부모를 찾습니다.
        """
        candidates = [lvl for lvl in active_parents if lvl < current_level]
        if not candidates:
            return None
        return active_parents[max(candidates)]

    # ─────────────────────────────────────────────────────────────
    #  헬퍼: context_path 생성
    # ─────────────────────────────────────────────────────────────
    def _build_context_path() -> str:
        """active_parents에 등록된 헤더 chunk들의 content를 레벨 오름차순으로 ' > '로 이어 붙입니다."""
        # 결과 리스트에서 active_parents의 chunk_id에 해당하는 content를 찾아 조합
        sorted_levels = sorted(active_parents.keys())
        parts: list[str] = []
        for lvl in sorted_levels:
            cid = active_parents[lvl]
            # result 리스트에서 해당 chunk_id의 content를 찾기
            for chunk in result:
                if chunk["chunk_id"] == cid:
                    parts.append(chunk["content"].strip())
                    break
        return " > ".join(parts)

    # ─────────────────────────────────────────────────────────────
    #  헬퍼: 버퍼 Flush → content 청크 생성
    # ─────────────────────────────────────────────────────────────
    def _flush_buffer():
        """누적된 content 버퍼를 하나의 content 청크로 확정하고 결과에 추가합니다."""
        nonlocal global_index, current_content_blocks, current_tokens, current_page_indices, current_state_type

        if not current_content_blocks:
            return

        # 텍스트 합치기 (블록 간 빈 줄 구분)
        raw_text = "\n\n".join(current_content_blocks)

        # context_path 기반 섹션 프리픽스 주입
        context_path = _build_context_path()
        if context_path:
            content_text = f"[Section: {context_path}]\n\n{raw_text}"
        else:
            content_text = raw_text

        chunk_id = str(uuid.uuid4())
        # Content 노드의 parent_id: 레벨을 무한대로 취급 → 가장 깊은 부모
        parent_id = _find_parent_id(MAX_LEVEL)

        unique_pages = sorted(list(set(current_page_indices))) if current_page_indices else []

        result.append({
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "global_index": global_index,
            "node_type": current_state_type,
            "text_level": None,
            "context_path": context_path,
            "content": content_text,
            "page_idx": unique_pages
        })
        global_index += 1

        # 버퍼 초기화
        current_content_blocks = []
        current_tokens = 0
        current_page_indices = []

    # ─────────────────────────────────────────────────────────────
    #  헬퍼: Header 청크 생성
    # ─────────────────────────────────────────────────────────────
    def _emit_header(text: str, level: int, page_idx: Optional[int] = None, node_type: str = "header"):
        """Header 블록을 청크로 확정하고 active_parents를 갱신합니다."""
        nonlocal global_index

        chunk_id = str(uuid.uuid4())

        # ── active_parents 스택 선행 정리 ──
        # 현재 레벨 이상의 깊은(큰) 키를 먼저 제거하여
        # context_path에 이미 만료된 하위 헤더가 포함되지 않도록 방지
        deeper_levels = [k for k in active_parents if k >= level]
        for k in deeper_levels:
            del active_parents[k]

        # parent_id: 현재 레벨보다 상위인 가장 깊은 부모
        parent_id = _find_parent_id(level)

        # context_path: 정리된 active_parents 기반 경로 + 자기 자신
        context_path = _build_context_path()
        if context_path:
            context_path = f"{context_path} > {text.strip()}"
        else:
            context_path = text.strip()

        pages = [page_idx] if page_idx is not None else []

        result.append({
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "global_index": global_index,
            "node_type": node_type,
            "text_level": level,
            "context_path": context_path,
            "content": text.strip(),
            "page_idx": pages
        })
        global_index += 1

        # ── active_parents에 자기 자신 등록 ──
        active_parents[level] = chunk_id

    # =================================================================
    #  메인 루프: content_list 단일 패스 순회
    # =================================================================
    for block in content_list:
        b_type = block.get("type", "")
        b_text = block.get("text", "").strip()
        b_level = block.get("text_level")
        b_page_idx = block.get("page_idx")

        if not b_text:
            continue

        # ─── node_type 판별 ───

        # [1] Original Header: text_level 키가 존재하고 정수형
        if b_level is not None and isinstance(b_level, int):
            # Header가 등장하면 버퍼에 쌓인 content를 무조건 Flush
            _flush_buffer()
            _emit_header(b_text, b_level, b_page_idx, node_type="header")
            current_state_type = "content"
            last_block_type = None
            continue

        # [2] Virtual Header & Independent Entities (Theorem, Definition, Proof, Example, etc.)
        label = ""
        rest_text = ""
        node_type = "content"
        header_level = None
        
        m_vh = virtual_header_re.match(b_text)
        if m_vh:
            label = m_vh.group(1).strip()
            rest_text = m_vh.group(2).strip()
            node_type = "definition" if "definition" in label.lower() else "theorem"
            header_level = config.virtual_header_level
        else:
            for e_type, e_regex in entity_patterns:
                m = e_regex.match(b_text)
                if m:
                    label = m.group(1).strip()
                    rest_text = m.group(2).strip()
                    node_type = e_type
                    header_level = config.entity_header_level
                    break
        
        if header_level is not None:    # 헤더이거나 독립엔티티인 경우
            _flush_buffer()
            
            # 헤더 승격
            _emit_header(label, header_level, b_page_idx, node_type=node_type)
            current_state_type = "content"
            last_block_type = None
            
            # 뒤쪽 본문 삽입 (Content로 처리됨)
            if rest_text:
                current_content_blocks.append(rest_text)
                current_tokens += count_tokens(rest_text)
                if b_page_idx is not None:
                    current_page_indices.append(b_page_idx)
                last_block_type = "text"
                    
            continue

        # [4] Content / Equation: 위 조건에 해당하지 않는 모든 블록
        new_tokens = count_tokens(b_text)
        is_equation = (b_type == "equation") or b_text.startswith("$$")
        current_block_type = "equation" if is_equation else "text"

        # 상태 머신 규칙: text -> text 연속 시 경계 분리
        if current_block_type == "text" and last_block_type == "text":
            _flush_buffer()
            current_state_type = "content"

        if is_equation:
            # 수식 바인딩: 직전 문단과 분리 금지 → SOFT_LIMIT 무시하고 기존 버퍼에 결합
            if current_tokens + new_tokens >= config.hard_limit:
                # 수식 폭탄 방어: HARD_LIMIT 초과 시 버퍼를 먼저 Flush 하고 수식을 새 버퍼에
                _flush_buffer()
                current_state_type = "content"
            # 수식을 현재 버퍼에 추가 (SOFT_LIMIT 초과해도 강제 결합)
            current_content_blocks.append(b_text)
            current_tokens += new_tokens
            if b_page_idx is not None:
                current_page_indices.append(b_page_idx)
        else:
            # 일반 텍스트: SOFT_LIMIT 도달 시 Flush 후 새 버퍼 시작
            if current_content_blocks and (current_tokens + new_tokens >= config.soft_limit):
                _flush_buffer()
                current_state_type = "content"
            current_content_blocks.append(b_text)
            current_tokens += new_tokens
            if b_page_idx is not None:
                current_page_indices.append(b_page_idx)
                
        last_block_type = current_block_type

    # ── 루프 종료 후 잔여 버퍼 Flush ──
    _flush_buffer()

    return result

# =============================================================================
# 전처리 전용 함수 (Data Rescue & Clean up)
# =============================================================================
def preprocess_mineru_data(content_list: list) -> list:
    """
    MinerU가 출력한 원시 블록 리스트에서 노이즈를 제거하고
    오분류된 데이터를 복원(구출)하는 전처리 함수입니다.

    처리 항목:
        1. header, page_number 타입 블록 드롭
        2. footer 중 오분류된 수식 구출 (직전 텍스트가 ':'로 끝나고 수식 기호 포함 시)
        3. image 타입 → 캡션 텍스트 블록으로 변환

    Args:
        content_list: MinerU 파싱 결과 원시 블록 리스트

    Returns:
        list: 노이즈가 제거된 정제 블록 리스트
    """
    cleaned_blocks = []

    for block in content_list:
        b_type = block.get('type')
        b_text = block.get('text', '')

        # 1. header, page_number 드롭
        if b_type in ('header', 'page_number'):
            continue

        # 2. 푸터 구출: 오분류된 수식 복원
        if b_type == 'footer':
            # 이전에 파싱된 가장 최근 텍스트 찾기
            prev_text = ""
            for prev_block in reversed(cleaned_blocks):
                if prev_block.get('text'):
                    prev_text = prev_block['text']
                    break

            # 콜론으로 끝나고 수식 관련 문자가 포함되어 있는지 검사
            if prev_text.strip().endswith(':'):
                if any(char in b_text for char in ['{', '}', '\\', '$']):
                    # 수식 오분류: 텍스트를 $$ $$로 감싸고 텍스트 타입으로 변경
                    block['text'] = f"$$ {b_text.strip()} $$"           # 임베딩 모델에서 $$ ~ $$를 적절하게 처리할 수 있는지 확인 후 필요에 따라 $ ~ $로 수정 요망
                    block['type'] = 'text'
                else:
                    continue  # 수식 기호가 없는 일반 푸터는 드롭
            else:
                continue  # 콜론으로 끝나지 않은 푸터도 제외

        # 3. 이미지 처리: 캡션을 텍스트 블록으로 변환
        if b_type == 'image':
            img_captions = block.get('image_caption', [])
            if isinstance(img_captions, list):
                caption_text = " ".join(str(c) for c in img_captions)
            else:
                caption_text = str(img_captions)

            block['text'] = f"[image caption: {caption_text}]"
            block['type'] = 'text'

        cleaned_blocks.append(block)

    return cleaned_blocks


# =============================================================================
# 메인 파이프라인: 전처리 → 시맨틱 청킹 → JSON 저장
# =============================================================================
if __name__ == "__main__":
    import sys

    # ── 인자 검증 ──
    if len(sys.argv) < 2:
        print("사용법: uv run python mineru_chunker.py <input.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    # ── 1. JSON 로드 ──
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"[1/3] 원시 블록 로드 완료: {len(raw_data)}개 블록")

    # ── 2. 전처리 (노이즈 제거 & 데이터 구출) ──
    cleaned_data = preprocess_mineru_data(raw_data)
    print(f"[2/3] 전처리 완료: {len(raw_data)}개 → {len(cleaned_data)}개 블록 ({len(raw_data) - len(cleaned_data)}개 드롭)")

    # ── 3. 시맨틱 청킹 ──
    chunked_data = chunk_mineru_math_doc(cleaned_data)
    print(f"[3/3] 시맨틱 청킹 완료: {len(chunked_data)}개 청크 생성")

    # ── 4. 결과 저장 (_chunked.json) ──
    output_path = input_path.with_name(f"{input_path.stem}_chunked.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장 완료: {output_path}")

