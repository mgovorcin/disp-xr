"""MkDocs hook for the published site (``mkdocs.public.yml``).

The published site leaves out the planning pages and some real-data
notebooks (``exclude_docs`` there). Links to those pages would break, so
this hook points them at the files on GitHub, and drops the gallery cards of
left-out notebooks.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

REPO_DOCS = "https://github.com/mgovorcin/disp-xr/blob/main/docs/"
LINK = re.compile(r"\]\(([^)#\s]+)(#[^)\s]*)?\)")
# One card of a "grid cards" list: the item line and its indented body.
CARD = re.compile(r"^-   [^\n]*\]\(([^)]+)\)[^\n]*\n(?:\n?    [^\n]*\n)*", re.MULTILINE)


def on_page_markdown(
    markdown: str, page: Any, config: Any, files: Any  # noqa: ARG001 (MkDocs API)
) -> str:
    """Rewrite links to left-out pages and drop cards of left-out notebooks."""
    excluded = {f.src_uri for f in files if f.inclusion.is_excluded()}
    folder = posixpath.dirname(page.file.src_uri)

    def target(link: str) -> str:
        return posixpath.normpath(posixpath.join(folder, link))

    def drop_card(match: re.Match) -> str:
        return "" if target(match.group(1)) in excluded else match.group(0)

    def to_github(match: re.Match) -> str:
        link, anchor = match.group(1), match.group(2) or ""
        if "://" in link or target(link) not in excluded:
            return match.group(0)
        return f"]({REPO_DOCS}{target(link)}{anchor})"

    markdown = CARD.sub(drop_card, markdown)
    return LINK.sub(to_github, markdown)
