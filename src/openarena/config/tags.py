# License Apache 2.0: (c) 2026 Atena-Reply

"""Tags: short lower-case labels on any registry entry, for browsing the lists.

Datasets carry topics (`math`, `science`, `multiple-choice`, `lang:en`),
models carry what they are (`openai`, `local`, `tools`, `128k-context`).
The rules are the same for every registry, and strict on purpose: two
spellings of one topic must not drift into two filters, and a stray
character is refused rather than silently mangled into a tag nobody can find
again. The app applies them on every read and write (`data_models.tags`);
`openarena check` reports a tag they refuse, with `suggest_tag`'s
spelling as the fix. The engine itself never reads tags.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator

from openarena.config.errors import Invalid

TAG_PATTERN = r"^[a-z0-9][a-z0-9:+./-]*$"
TAG_MAX_LENGTH = 40
MAX_TAGS = 32


def normalize_tag(raw: str) -> str:
    """One tag as stored: trimmed, lower-case, spaces and `_` folded to `-`.

    Raises `ValueError` for anything that still does not fit `TAG_PATTERN`,
    so a stray character is refused rather than silently mangled into a tag
    nobody can find again.
    """
    tag = re.sub(r"[\s_]+", "-", str(raw).strip().lower())
    if not tag:
        raise ValueError("a tag cannot be empty")
    if len(tag) > TAG_MAX_LENGTH:
        raise ValueError(f"tag {raw!r} is longer than {TAG_MAX_LENGTH} characters")
    if not re.fullmatch(TAG_PATTERN, tag):
        raise ValueError(
            f"tag {raw!r} is invalid: letters, digits, '-', '.', ':', '+' and '/' only, "
            "starting with a letter or digit"
        )
    return tag


def normalize_tags(tags: list[str]) -> list[str]:
    """`normalize_tag` over a list, blanks dropped and duplicates removed in order."""
    out: list[str] = []
    for raw in tags:
        if not str(raw).strip():
            continue
        tag = normalize_tag(raw)
        if tag not in out:
            out.append(tag)
    if len(out) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags")
    return out


def lenient_tags(candidates: list[str]) -> list[str]:
    """Normalize what can be normalized and drop the rest: for suggestions, not user input."""
    out: list[str] = []
    for raw in candidates:
        try:
            tag = normalize_tag(raw)
        except ValueError:
            continue
        if tag not in out:
            out.append(tag)
    return out[:MAX_TAGS]


def suggest_tag(raw: str) -> str | None:
    """The nearest tag `normalize_tag` accepts, for a fix to offer; never applied to what is stored.

    Accents stripped, every other character outside the pattern folded to
    `-`, cut to the maximum length. None when nothing is left.
    """
    text = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    tag = re.sub(r"[\s_]+", "-", text.strip().lower())
    tag = re.sub(r"[^a-z0-9:+./-]+", "-", tag)
    tag = re.sub(r"-{2,}", "-", tag).strip("-:+./")
    tag = tag[:TAG_MAX_LENGTH].rstrip("-:+./")
    return tag or None


def _tag(raw: Any) -> str:
    try:
        return normalize_tag(raw)
    except ValueError as e:
        fixed = suggest_tag(raw)
        raise Invalid(str(e), f"write it as `{fixed}`" if fixed else f"remove {raw!r}; a tag needs a letter or a digit") from None


def _drop_blank(value: Any) -> Any:
    if isinstance(value, list):
        return [t for t in value if str(t).strip()]
    if isinstance(value, str):
        fixed = suggest_tag(value)
        raise Invalid("`tags:` is a list", f"write it as a list: `tags: [{fixed}]`" if fixed else "write it as a list: `tags: [math]`")
    return value


def _unique(tags: list[str]) -> list[str]:
    out = list(dict.fromkeys(tags))
    if len(out) > MAX_TAGS:
        raise Invalid(f"more than {MAX_TAGS} tags", f"keep at most {MAX_TAGS}; remove {len(out) - MAX_TAGS}")
    return out


# One tag as stored: refused, with the spelling that would do, when the rule does not take it.
Tag = Annotated[str, BeforeValidator(_tag)]
# `tags:` on any entry: blanks dropped, each a `Tag`, duplicates removed in order, at most `MAX_TAGS`.
Tags = Annotated[list[Tag], BeforeValidator(_drop_blank), AfterValidator(_unique)]

