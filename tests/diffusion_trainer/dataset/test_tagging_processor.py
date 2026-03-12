from pathlib import Path

from PIL import Image

from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor


def test_import_sidecar_txt_tags(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    target_dir = tmp_path / "metadata" / "tags"
    image_dir.mkdir(parents=True)

    image_path = image_dir / "sample.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    image_path.with_suffix(".txt").write_text("1girl, solo, smile, solo", encoding="utf-8")

    processor = TaggingProcessor(
        img_path=str(image_dir),
        target_path=str(target_dir),
        num_workers=1,
        skip_existing=False,
        tag_source="sidecar_txt",
    )

    processor()

    output_path = processor.get_tag_save_path(image_path)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "1girl, solo, smile"
