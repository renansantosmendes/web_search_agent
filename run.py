"""
Simple deepagents (LangChain) agent with ChatOpenAI, a Tavily search tool,
loaded skills, and a terminal chat loop.

Project layout:
    agente/
    ├── agente_chat.py
    └── skills/
        └── tavily-research/
            └── SKILL.md

Installation:
    pip install deepagents langchain-openai tavily-python

Required environment variables:
    export OPENAI_API_KEY="your-openai-key"
    export TAVILY_API_KEY="your-tavily-key"

Usage:
    python agente_chat.py
"""

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from tools.web_search import web_search
from tools.scrape import scrape_page

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


BASE_DIR = Path(__file__).resolve().parent


def create_agent():
    """Create the agent with ChatOpenAI, the Tavily search tool, and the skills folder."""
    model = ChatOpenAI(model="gpt-4o", temperature=0)

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


def chat(agent):
    """Simple chat loop: read user input, send it to the agent, and print the response."""
    history = []

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "sair"):
            print("Ending chat.")
            break

        history.append({"role": "user", "content": user_input})

        response = agent.invoke({"messages": history})
        history = response["messages"]

        text = history[-1].content
        print(f"Agent: {text}\n")


if __name__ == "__main__":
    agent = create_agent()
    chat(agent)