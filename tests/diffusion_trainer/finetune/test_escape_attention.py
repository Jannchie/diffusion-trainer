"""Escaping keeps booru tags literal under the A1111-style attention parser."""

import pytest

from diffusion_trainer.finetune.base import escape_attention_syntax


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ask_(askzy)", r"ask_\(askzy\)"),
        ("okita_souji_(fate)", r"okita_souji_\(fate\)"),
        ("1girl, solo, red_eyes", "1girl, solo, red_eyes"),
        ("[bracketed]", r"\[bracketed\]"),
        ("back\\slash", "back\\\\slash"),
    ],
)
def test_escape_attention_syntax(raw: str, expected: str) -> None:
    assert escape_attention_syntax(raw) == expected


def test_escaped_tags_parse_as_literal_text() -> None:
    """Round-trip through the enhanced embedder's parser: literal text, weight 1.0."""
    parser = pytest.importorskip("diffusion_prompt_embedder.core.parser")

    prompt = "1girl, ask_(askzy), okita_souji_(fate), red_eyes"
    parsed = parser.parse_prompt_attention(escape_attention_syntax(prompt))
    assert parsed == [[prompt, 1.0]]
