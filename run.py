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


from dotenv import load_dotenv

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

load_dotenv()  # Load environment variables from .env file


from agent import create_agent


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

        start = len(history)
        response = agent.invoke({"messages": history})
        history = response["messages"]

        for message in history[start:]:
            if message.type == "tool":
                print(f"[tool {message.name}]\n{message.content}\n")

        text = history[-1].content
        print(f"Agent: {text}\n")


if __name__ == "__main__":
    agent = create_agent()
    chat(agent)
