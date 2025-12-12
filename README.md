# CM ReAct Agent with RAG & Web Search

This folder contains a lightweight ReAct agent that speaks the OpenAI chat-completions API and two tools wired for tool-calling:

- `rag_search`: local retrieval over your `.txt` corpus, indexed with **pyserini**.

## Quick start
1. Install deps (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
2. Copy env template and fill in your values:
   ```
   cp .env_example .env
   # Edit .env with your keys:
   # OPENAI_API_KEY=...
   # OPENAI_BASE_URL=...      # e.g. https://api.openai.com/v1 or your proxy
   # OPENAI_MODEL=...         # e.g. gpt-4o-mini
   # RAG_INDEX_DIR=/home/zlp/CM/rag/index  # optional override
   ```
3. Prepare RAG data:
   - Drop raw `.txt` files into `rag/raw_docs/`.
   - Build the Lucene index (creates `rag/json_collection` + `rag/index`):
     ```
     python rag/build_index.py --input rag/raw_docs --index rag/index --collection rag/json_collection
     ```
4. Run an example query:
   ```
   python -m CM.main
   ```

## Files
- `agent/llm_client.py`: OpenAI-compatible async client (base URL & model left for you to fill via env).
- `agent/react_agent.py`: ReAct loop with tool-calling.
- `tools/rag_tool.py`: Pyserini-backed retriever over local corpus.
- `tools/web_search.py`: Serper-based web search (set `SERPER_API_KEY`).
- `rag/build_index.py`: turns `.txt` corpus into a Lucene index for RAG.

## Notes
- The agent registers tools at startup; add more tools under `tools/` and register them in `main.py`.
- Pyserini requires Java; ensure `JAVA_HOME` is set if indexing errors appear.
- If you prefer a different search provider, swap `web_search.py` to use your API of choice and expose the key via env/config.

