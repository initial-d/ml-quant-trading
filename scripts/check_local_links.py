"""Check that local Markdown links resolve to files in this checkout."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


LINK = re.compile(r"\[[^\]\n]+\]\(<([^>]+)>\)|\[[^\]\n]+\]\(([^\s)]+)(?:\s+[^)]*)?\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^]]+\]:\s*(?:<([^>]+)>|([^\s]+))", re.MULTILINE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def destinations(source: str):
    """Yield inline and reference destinations outside fenced code blocks."""
    lines = []
    fence_char = None
    fence_length = 0
    for line in source.splitlines():
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = None
            lines.append("")
        else:
            lines.append("" if fence_char else line)
    visible = "\n".join(lines)
    for match in LINK.finditer(visible):
        yield match.group(1) or match.group(2)
    for match in REFERENCE.finditer(visible):
        yield match.group(1) or match.group(2)


def broken_links(root: Path):
    for source in [*root.glob("*.md"), *root.glob("docs/**/*.md")]:
        for destination in destinations(source.read_text(encoding="utf-8")):
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or destination.startswith(("#", "mailto:", "//")):
                continue
            path = unquote(parsed.path)
            if not path:
                continue
            target = (root / path.lstrip("/")) if path.startswith("/") else (source.parent / path)
            if not target.exists():
                yield f"{source.relative_to(root)}: {destination}"


if __name__ == "__main__":
    repository = Path(__file__).resolve().parent.parent
    failures = list(broken_links(repository))
    if failures:
        print("Broken local Markdown links:\n" + "\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("Local Markdown links resolve")
