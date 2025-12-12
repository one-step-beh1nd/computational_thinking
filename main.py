import asyncio
from .agent import LLMClient, ReactAgent
from .tools import RagSearchTool, ToolRegistry


async def main() -> None:
    registry = ToolRegistry()
    registry.register(RagSearchTool())
    
    agent = ReactAgent(llm_client=LLMClient(), tool_registry=registry, max_turns=8)

    # 开场白仅展示，不加入对话历史
    print("你好，我是你的智能讲解助手，随时为你解析从深蓝到 AlphaGo 的棋类 AI 发展，也能解答各种其他问题。\n")
    print("小组成员：赵乐朋\n")
    print("请输入问题，输入空行或 Ctrl+C 结束。\n")

    loop = asyncio.get_event_loop()
    while True:
        try:
            query = await loop.run_in_executor(None, input, ">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not query.strip():
            print("已退出。")
            break

        answer = await agent.run(query.strip())
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())

