"""Creates the deep agent with ChatOpenAI, the web tools and the skills folder.

Shared by the terminal chat (run.py) and the HTTP API (app.py).

Required environment variables:
    OPENAI_API_KEY   key of the OpenAI model
    TAVILY_API_KEY   key of the Tavily search API
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_openai import ChatOpenAI

from tools.scrape import scrape_page
from tools.web_search import web_search

BASE_DIR = Path(__file__).resolve().parent


def create_agent():
    """Create the agent with ChatOpenAI, the web tools, and the skills folder.

    Returns
    -------
    object
        A compiled deep agent; call ``invoke({"messages": [...]})`` on it.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_deep_agent(
        model=model,
        tools=[web_search, scrape_page],
        backend=FilesystemBackend(root_dir=str(BASE_DIR), virtual_mode=True),
        skills=[str(BASE_DIR / "skills")],
        system_prompt=(
            "You are a helpful assistant. Answer clearly and concisely. "
            "When you need up-to-date information, use the web_search tool "
            "and cite the source URLs in your answer."
        ),
    )
    return agent
