import asyncio
import sys

from agent import LLMClient, ReactAgent
from tools import RagSearchTool, ToolRegistry, WebSearchTool


async def main(query: str) -> None:
    registry = ToolRegistry()
    registry.register(RagSearchTool())
    registry.register(WebSearchTool())

    agent = ReactAgent(llm_client=LLMClient(), tool_registry=registry, max_turns=8)
    answer = await agent.run(query)
    print("Answer:", answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py '<your question>'")
    asyncio.run(main(sys.argv[1]))

