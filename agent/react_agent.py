from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from .llm_client import LLMClient
from .types import Message
from ..tools.registry import ToolRegistry


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
        messages: List[Message] = [
            Message(
                role="system",
                content=(
                    "You are a helpful ReAct agent. Think step by step. "
                    "If tools are useful, call them using the available functions. "
                    "When you are ready to answer, reply directly to the user."
                ),
            ),
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

        return "Reached maximum iterations without a final answer."

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

