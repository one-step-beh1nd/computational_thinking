from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult


class ToolRegistry:
    """Simple in-memory tool registry."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_function_definitions(self) -> List[dict]:
        return [tool.to_function() for tool in self._tools.values()]

    async def execute(self, name: str, inputs: Dict) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        try:
            return await tool(**inputs)
        except Exception as exc:  # pragma: no cover - guardrail
            return ToolResult(success=False, error=str(exc))

