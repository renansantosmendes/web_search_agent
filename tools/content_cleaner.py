"""Cleans the markdown of a page so that the text reads without the page chrome.

Images are removed, links are replaced by their text, and navigation links to
site sections (such as ``[Economia](/economia)``) are dropped. Runs of blank
lines are collapsed into one.
"""

import re

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
SECTION_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(/[^/)\s]*\)")
LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def clean_markdown(text: str) -> str:
    """Remove images, navigation links and extra blank lines from markdown.

    Parameters
    ----------
    text : str
        Markdown of a page, as returned by the search API in
        ``raw_content``.

    Returns
    -------
    str
        The text with links replaced by their labels and without images.
        An empty string if nothing is left.

    Example
    -------
    >>> clean_markdown("[Economia](/economia) [Dólar sobe](/noticia/dolar)")
    'Dólar sobe'
    """
    text = IMAGE_PATTERN.sub("", text)
    text = SECTION_LINK_PATTERN.sub("", text)
    text = LINK_PATTERN.sub(r"\1", text)

    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        lines.append(line)

    return "\n".join(lines).strip()
