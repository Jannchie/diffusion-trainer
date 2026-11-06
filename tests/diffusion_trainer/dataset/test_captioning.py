"""The MOSS recaptioning contract: what gets accepted, what gets retried."""

import json

import pytest

from diffusion_trainer.dataset.captioning import CaptionClient, CaptionValidationError, build_prompt, build_reference, parse_caption_response

VALID = {
    "caption": {"long": "A girl under cherry blossoms.", "medium": "A girl, blossoms.", "short": "A girl."},
    "watermark": [{"box_2d": [950, 10, 990, 200], "label": "@artist"}],
    "borders": ["none"],
}


def test_parses_a_reply_with_inline_reasoning_and_a_fence() -> None:
    reply = "<think>looking at the image... {not json}</think>\n```json\n" + json.dumps(VALID) + "\n```"
    record = parse_caption_response(reply)

    assert record.short == "A girl."
    assert record.watermark[0].box_2d == [950, 10, 990, 200]
    assert record.borders == ["none"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["caption"].update(short="  "),
        lambda d: d["caption"].pop("medium"),
        lambda d: d.update(watermark=[{"box_2d": [0, 0, 1200, 10], "label": "x"}]),
        lambda d: d.update(watermark=[{"box_2d": [0, 0, 10], "label": "x"}]),
        lambda d: d.update(borders=["none", "left"]),
        lambda d: d.update(borders=["top"]),
    ],
)
def test_schema_violations_are_rejected(mutate) -> None:  # noqa: ANN001
    data = json.loads(json.dumps(VALID))
    mutate(data)
    with pytest.raises(CaptionValidationError):
        parse_caption_response(json.dumps(data))


def test_missing_watermark_and_borders_default_to_none() -> None:
    record = parse_caption_response(json.dumps({"caption": VALID["caption"]}))
    assert record.watermark == []
    assert record.borders == ["none"]


def test_reference_groups_tags_and_drops_synthetic_ones() -> None:
    reference = build_reference(
        ["best quality", "newest", "wlop", "genshin_impact", "lumine_(genshin_impact)", "1girl", "long_hair", "highres"],
        ["quality", "year", "artist", "copyright", "character", "general", "general", "meta"],
    )
    assert reference.splitlines() == [
        "Artist: wlop",
        "Series: genshin impact",
        "Characters: lumine (genshin impact)",
        "Tags: 1girl, long hair",
    ]


def test_prompt_and_request_follow_the_user_guide() -> None:
    prompt = build_prompt("Tags: 1girl")
    assert prompt.endswith("Here is the original caption reference:\nTags: 1girl")
    body = CaptionClient(endpoint="http://x/v1", model="m").request_body(prompt, "data:image/jpeg;base64,AA==")
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (1.0, 0.95, 16384)
