"""Booru tag spelling shared by prompt assembly and the recaption reference."""

# Kaomoji whose underscore is part of the face (the WD14 tagger's list): they
# keep it when every other tag has its underscores read as spaces.
KAOMOJI = frozenset(
    {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"},
)


def booru_tag_to_text(tag: str) -> str:
    """``long_hair`` -> ``long hair``; kaomoji such as ``^_^`` stay as they are."""
    return tag if tag in KAOMOJI else tag.replace("_", " ")
