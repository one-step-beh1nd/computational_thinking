from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from .llm_client import LLMClient
from .types import Message
from ..tools.registry import ToolRegistry
# from CM.tools.registry import ToolRegistry



class ReactAgent:
    """
    Minimal ReAct-style agent with OpenAI tool-calling.
    - Builds a short system prompt with tool usage instructions.
    - Loops until the LLM returns a final answer or max turns reached.
    - Executes declared tool calls through ToolRegistry.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_turns: int = 8,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.tools = tool_registry or ToolRegistry()
        self.max_turns = max_turns

    async def run(self, query: str) -> str:
        system_prompt = (
            "你是一名清晰、耐心且善于讲解的智能助理。你特别擅长讲述棋类人工智能的发展"
            "（从深蓝到 AlphaGo）以及其中体现的计算思维，包括：如何用算法、分解、抽象、搜索、"
            "模式识别与评价函数来解决复杂问题。"
            "同时，你也是一个通用型助理，能够回答任何领域的普通问题。\n\n"
            "在回答时：\n"
            "- 优先给出结构化、易理解的解释。\n"
            "- 面向不同背景的用户调整表达方式。\n"
            "- 不捏造事实，不暴露内部推理链。\n"
            "- 遇到不确定的问题，明确说明并给出合理的方向。"
        )

        messages: List[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]

        for _ in range(self.max_turns):
            llm_message = await self.llm.complete(
                messages,
                tools=self.tools.get_function_definitions(),
                tool_choice="auto",
            )
            messages.append(Message.from_openai(llm_message))

            tool_calls = llm_message.get("tool_calls") or []
            if tool_calls:
                tool_results = await asyncio.gather(
                    *[
                        self._execute_tool_call(call)
                        for call in tool_calls
                    ]
                )
                messages.extend(tool_results)
                continue

            # Final answer from assistant
            final_content = llm_message.get("content") or ""
            return final_content

        # 达到最大轮次后仍无明确回答：直接向 LLM 请求最终答复（不再调用工具）
        fallback = await self.llm.complete(messages, tools=None, tool_choice=None)
        messages.append(Message.from_openai(fallback))
        return fallback.get("content") or "这是我的直接回答：目前根据已知信息做出的最佳解读。"

    async def _execute_tool_call(self, tool_call: dict) -> Message:
        """Execute a single tool call and wrap its result as a tool message."""
        name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments") or "{}"
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = {"input": raw_args}

        result = await self.tools.execute(name, parsed_args)
        if hasattr(result, "render_for_llm"):
            content = result.render_for_llm()
        else:
            content = json.dumps(result, ensure_ascii=False)

        return Message.format_tool_result(
            tool_name=name,
            tool_call_id=tool_call.get("id") or "",
            result=content,
        )

