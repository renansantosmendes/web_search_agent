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
            "Você é um assistente de pesquisa na web. Responda sempre em "
            "português do Brasil, de forma clara e concisa. "
            "Para informações atualizadas, use a ferramenta web_search e, se "
            "precisar do conteúdo completo de uma página, use scrape_page. "
            "Cite as URLs de origem na resposta. "
            "Se a pergunta for sobre você, o que você faz ou quais skills tem, "
            "responda diretamente com esta descrição, sem procurar em arquivos. "
            "Você tem a skill tavily-research, que orienta como conduzir uma "
            "pesquisa com a API da Tavily."
        ),
    )
    return agent
