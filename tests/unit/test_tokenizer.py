"""Unit tests for latex_regex_tokenizer — 토크나이징 검증."""

import pytest

from theoremlens.pipeline.retriever import latex_regex_tokenizer
from theoremlens.config import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATTERN = config.bm25_token_pattern


# ---------------------------------------------------------------------------
# TestLatexTokenizer
# ---------------------------------------------------------------------------


class TestLatexTokenizer:
    """latex_regex_tokenizer의 토크나이징 동작을 검증하는 테스트 스위트."""

    # ── 기본 동작 ────────────────────────────────────────────────────────────

    def test_empty_string_returns_empty_list(self):
        """빈 문자열 입력 시 빈 리스트를 반환해야 한다."""
        assert latex_regex_tokenizer("") == []

    def test_plain_text_lowercased(self):
        """일반 영문 텍스트는 소문자로 변환되어야 한다."""
        tokens = latex_regex_tokenizer("Hello World")
        assert "hello" in tokens
        assert "world" in tokens
        # 원본 대문자 그대로는 없어야 함
        assert "Hello" not in tokens
        assert "World" not in tokens

    def test_digits_preserved(self):
        """숫자 토큰이 올바르게 추출되어야 한다."""
        tokens = latex_regex_tokenizer("x^2 + 10 = 12")
        assert "2" in tokens
        assert "10" in tokens
        assert "12" in tokens

    # ── LaTeX 커맨드 보존 ────────────────────────────────────────────────────

    def test_latex_fraction_preserved(self):
        r"""\\frac 커맨드가 원형 그대로 토큰에 포함되어야 한다."""
        tokens = latex_regex_tokenizer(r"\frac{a}{b}")
        assert r"\frac" in tokens

    def test_latex_greek_letters_preserved(self):
        r"""그리스 문자 커맨드(\\alpha, \\beta, \\gamma)가 보존되어야 한다."""
        text = r"\alpha + \beta = \gamma"
        tokens = latex_regex_tokenizer(text)
        assert r"\alpha" in tokens
        assert r"\beta" in tokens
        assert r"\gamma" in tokens

    def test_latex_operators_preserved(self):
        r"""\\sum, \\int, \\prod, \\lim 등 주요 연산자 커맨드가 보존되어야 한다."""
        text = r"\sum_{i=1}^{n} \int_0^1 \prod_{k} \lim_{x \to 0}"
        tokens = latex_regex_tokenizer(text)
        for cmd in [r"\sum", r"\int", r"\prod", r"\lim"]:
            assert cmd in tokens, f"Expected token '{cmd}' not found"

    def test_latex_inequality_symbols_preserved(self):
        r"""\\le, \\ge, \\neq, \\approx 등 관계 연산자가 보존되어야 한다."""
        text = r"x \le y \ge z, a \neq b, c \approx d"
        tokens = latex_regex_tokenizer(text)
        for cmd in [r"\le", r"\ge", r"\neq", r"\approx"]:
            assert cmd in tokens, f"Expected token '{cmd}' not found"

    def test_latex_commands_not_lowercased(self):
        r"""LaTeX 커맨드(백슬래시 시작)는 소문자 변환 없이 원형을 유지해야 한다."""
        tokens = latex_regex_tokenizer(r"\Phi \Psi \Omega")
        assert r"\Phi" in tokens
        assert r"\Psi" in tokens
        assert r"\Omega" in tokens
        # 소문자화된 버전은 없어야 함
        assert r"\phi" not in tokens

    # ── 혼합 수식 ────────────────────────────────────────────────────────────

    def test_mixed_math_text(self):
        r"""일반 텍스트와 LaTeX 혼합 입력에서 양쪽 토큰이 모두 추출되어야 한다."""
        text = r"Let $x \in V$ be a vector in an inner product space."
        tokens = latex_regex_tokenizer(text)
        assert "let" in tokens
        assert "inner" in tokens
        assert "product" in tokens
        assert "space" in tokens
        assert r"\in" in tokens

    def test_integral_expression(self):
        r"""\\int_0^1 f(t) dt 표현에서 \\int 커맨드가 추출되어야 한다."""
        text = r"We calculate \int_0^1 f(t) dt using integration by parts."
        tokens = latex_regex_tokenizer(text)
        assert r"\int" in tokens
        assert "integration" in tokens
        assert "by" in tokens
        assert "parts" in tokens

    # ── 토큰 수 및 중복 ──────────────────────────────────────────────────────

    def test_repeated_tokens_included(self):
        """동일 단어가 여러 번 등장하면 모두 토큰 리스트에 포함되어야 한다."""
        text = "the matrix and the vector form the basis"
        tokens = latex_regex_tokenizer(text)
        assert tokens.count("the") >= 3

    def test_returns_list_type(self):
        """반환 타입이 list여야 한다."""
        result = latex_regex_tokenizer("test input")
        assert isinstance(result, list)

    def test_all_tokens_are_strings(self):
        """반환된 모든 토큰이 str 타입이어야 한다."""
        tokens = latex_regex_tokenizer(r"Let \alpha be a scalar and x a vector.")
        assert all(isinstance(t, str) for t in tokens)

    # ── 커스텀 패턴 ──────────────────────────────────────────────────────────

    def test_custom_pattern_overrides_default(self):
        """custom pattern 인자가 주어지면 해당 패턴으로 토크나이징 되어야 한다."""
        simple_pattern = r"[a-zA-Z]+"
        tokens = latex_regex_tokenizer(r"\alpha + x = 1", simple_pattern)
        assert "alpha" in tokens
        assert "x" in tokens
        assert "1" not in tokens

    # ── 특수 문자 엣지 케이스 ────────────────────────────────────────────────

    def test_punctuation_handled(self):
        """구두점(., ,, ;)이 포함된 텍스트에서도 올바르게 토크나이징 되어야 한다."""
        tokens = latex_regex_tokenizer("Therefore, x equals 5.")
        assert "therefore" in tokens
        assert "x" in tokens
        assert "equals" in tokens
        assert "5" in tokens

    def test_whitespace_only_returns_empty(self):
        """공백만 있는 문자열은 빈 리스트를 반환해야 한다."""
        tokens = latex_regex_tokenizer("   \t\n  ")
        assert tokens == []

    def test_multiline_text(self):
        """여러 줄에 걸친 텍스트도 올바르게 처리되어야 한다."""
        text = "Line one with \\alpha\nLine two with \\beta\nLine three"
        tokens = latex_regex_tokenizer(text)
        assert r"\alpha" in tokens
        assert r"\beta" in tokens
        assert "line" in tokens
        assert "one" in tokens
        assert "two" in tokens
        assert "three" in tokens
