from __future__ import annotations

import os
from typing import Any, Dict, List

from pyserini.search.lucene import LuceneSearcher

from .base import BaseTool, ToolResult


class RagSearchTool(BaseTool):
    """
    Local RAG search over a Lucene index built by `rag/build_index.py`.

    Environment variables:
      - RAG_INDEX_DIR (defaults to /home/zlp/CM/rag/index)
    """

    name = "rag_search"
    description = "Search the local RAG index built from .txt files."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to retrieve supporting passages.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of hits to return",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    async def __call__(self, query: str, top_k: int = 3, **_: Any) -> ToolResult:
        index_dir = os.getenv("RAG_INDEX_DIR", "/home/zlp/CM/rag/index")
        if not os.path.isdir(index_dir):
            return ToolResult(
                success=False,
                error=f"RAG index not found at {index_dir}. Run rag/build_index.py first.",
            )

        searcher = LuceneSearcher(index_dir)
        hits = searcher.search(query, k=top_k)

        results: List[Dict[str, Any]] = []
        for hit in hits:
            results.append(
                {
                    "id": hit.docid,
                    "score": hit.score,
                    "contents": hit.raw or hit.contents,
                }
            )

        return ToolResult(success=True, output=results)

