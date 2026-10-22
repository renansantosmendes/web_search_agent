"""
Tavily web search tool for the agent.

The tool name ("web_search") must match the one mentioned in
skills/tavily-research/SKILL.md.
"""

import os

from langchain_core.tools import tool
from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

MAX_RAW_CHARS = 20000


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web with Tavily and return a short answer plus relevant sources.

    Use this tool when the user asks about recent events, current facts,
    prices, roles or anything that may have changed, or when sources are needed.

    Args:
        query: Short search query (3 to 6 keywords work best).
        max_results: Number of results to return (default 5).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY is not set"}

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content="markdown",
    )

    return {
        "answer": response.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "raw_content": (r.get("raw_content") or "")[:MAX_RAW_CHARS],
            }
            for r in response.get("results", [])
        ],
    }