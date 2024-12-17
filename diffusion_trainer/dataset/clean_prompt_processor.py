"""Clean tags and caption in metadata.json file."""

import argparse
import json
import re
from pathlib import Path

from diffusion_trainer.shared import get_progress

TAGS_EXT = ".txt"
CAPTION_EXT = ".caption"

PATTERN_HAIR_LENGTH = re.compile(r", (long|short|medium) hair, ")
PATTERN_HAIR_CUT = re.compile(r", (bob|hime) cut, ")
PATTERN_HAIR = re.compile(r", ([\w\-]+) hair, ")
PATTERN_WORD = re.compile(r", ([\w\-]+|hair ornament), ")


PATTERNS_REMOVE_IN_MULTI = [
    PATTERN_HAIR_LENGTH,
    PATTERN_HAIR_CUT,
    re.compile(r", [\w\-]+ eyes, "),
    re.compile(r", ([\w\-]+ sleeves|sleeveless), "),
    re.compile(
        r", (ponytail|braid|ahoge|twintails|[\w\-]+ bun|single hair bun|single side bun|two side up|two tails|[\w\-]+ braid|sidelocks), ",
    ),
]

CAPTION_REPLACEMENTS = [
    ("anime anime", "anime"),
    ("young ", ""),
    ("anime girl", "girl"),
    ("cartoon female", "girl"),
    ("cartoon lady", "girl"),
    ("cartoon character", "girl"),
    ("cartoon woman", "girl"),
    ("cartoon women", "girls"),
    ("cartoon girl", "girl"),
    ("anime female", "girl"),
    ("anime lady", "girl"),
    ("anime character", "girl"),
    ("anime woman", "girl"),
    ("anime women", "girls"),
    ("lady", "girl"),
    ("female", "girl"),
    ("woman", "girl"),
    ("women", "girls"),
    ("people", "girls"),
    ("person", "girl"),
    ("a cartoon figure", "a figure"),
    ("a cartoon image", "an image"),
    ("a cartoon picture", "a picture"),
    ("an anime cartoon image", "an image"),
    ("a cartoon anime drawing", "a drawing"),
    ("a cartoon drawing", "a drawing"),
    ("girl girl", "girl"),
]


def clean_tags(tags: str) -> str:
    """Clean tags with some replacements."""
    tags = tags.replace("^_^", "^@@@^")
    tags = tags.replace("_", " ")
    tags = tags.replace("^@@@^", "^_^")

    tags = ", " + tags.replace(", ", ", , ") + ", "

    if "girls" in tags or "boys" in tags:
        for pat in PATTERNS_REMOVE_IN_MULTI:
            found = pat.findall(tags)
            if len(found) > 1:
                tags = pat.sub("", tags)

        srch_hair_len = PATTERN_HAIR_LENGTH.search(tags)
        if srch_hair_len:
            org = srch_hair_len.group()
            tags = PATTERN_HAIR_LENGTH.sub(", @@@, ", tags)

        found = PATTERN_HAIR.findall(tags)
        if len(found) > 1:
            tags = PATTERN_HAIR.sub("", tags)

        if srch_hair_len:
            org = srch_hair_len.group()
            tags = tags.replace(", @@@, ", org)

    found = PATTERN_WORD.findall(tags)
    for word in found:
        if re.search(rf", ((\w+) )+{word}, ", tags):
            tags = tags.replace(f", {word}, ", "")

    tags = tags.replace(", , ", ", ")
    if tags.startswith(", ") and tags.endswith(", "):
        return tags[2:-2]
    return tags


def clean_caption(caption: str) -> str:
    """Clean caption with some replacements."""
    for rf, rt in CAPTION_REPLACEMENTS:
        replaced = True
        while replaced:
            bef = caption
            caption = caption.replace(rf, rt)
            replaced = bef != caption
    return caption


class CleanPromptProcessor:
    """Clean tags and caption in metadata.json file."""

    def __init__(self, meta_path: str | Path) -> None:
        """Initialize."""
        self.meta_path = Path(meta_path)
        with self.meta_path.open(encoding="utf-8") as file:
            self.metadata = json.load(file)

    def __call__(self) -> None:
        """Clean tags and caption."""
        with get_progress() as progress:
            task = progress.add_task("Cleaning Prompts...", total=len(self.metadata))
            for key in self.metadata:
                if tags := self.metadata[key].get("tags"):
                    self.metadata[key]["tags"] = clean_tags(tags)
                if caption := self.metadata[key].get("caption"):
                    self.metadata[key]["caption"] = clean_caption(caption)
                progress.update(task, advance=1)

        with self.meta_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    args = parser.parse_args()
    cleaner = CleanPromptProcessor(args.meta_path)
    cleaner()
