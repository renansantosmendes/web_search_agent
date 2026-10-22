"""
Web scraping tool for the agent.

The tool name ("scrape_page") must match the one mentioned in
skills/web-scrape/SKILL.md.

Requires: pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

MAX_CHARS = 8000
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; agente-chat/1.0)"


@tool
def scrape_page(url: str) -> dict:
    """Download a web page and return its title and main text content.

    Use this tool when the user gives a URL or asks for the content of a
    specific page. Do not use it for general questions; use web_search instead.

    Args:
        url: Full URL starting with http:// or https://.
    """
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Could not fetch page: {exc}"}

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove tags that rarely contain the main content.
    noise_tags = ["script", "style", "noscript", "nav", "footer", "header", "aside"]
    for tag in soup(noise_tags):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = " ".join(soup.get_text(separator=" ").split())

    return {
        "url": url,
        "title": title,
        "content": text[:MAX_CHARS],
        "truncated": len(text) > MAX_CHARS,
    }