from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def txt_to_json_collection(src_dir: Path, collection_dir: Path) -> List[Path]:
    collection_dir.mkdir(parents=True, exist_ok=True)
    doc_paths: List[Path] = []
    for txt_path in sorted(src_dir.glob("*.txt")):
        doc = {"id": txt_path.stem, "contents": txt_path.read_text(encoding="utf-8", errors="ignore")}
        out_path = collection_dir / f"{txt_path.stem}.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        doc_paths.append(out_path)
    return doc_paths


def build_lucene_index(collection_dir: Path, index_dir: Path, threads: int = 4) -> None:
    """
    Call pyserini's CLI to build a Lucene index.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "-collection",
        "JsonCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(threads),
        "-input",
        str(collection_dir),
        "-index",
        str(index_dir),
        "-storePositions",
        "-storeRaw",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pyserini Lucene index from .txt files.")
    parser.add_argument("--input", type=Path, default=Path("/home/zlp/CM/rag/raw_docs"), help="Folder of raw .txt files")
    parser.add_argument("--collection", type=Path, default=Path("/home/zlp/CM/rag/json_collection"), help="Where to write JsonCollection docs")
    parser.add_argument("--index", type=Path, default=Path("/home/zlp/CM/rag/index"), help="Output index directory")
    parser.add_argument("--threads", type=int, default=4, help="Indexing threads")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input dir not found: {args.input}")

    docs = txt_to_json_collection(args.input, args.collection)
    if not docs:
        raise SystemExit(f"No .txt files found under {args.input}")

    build_lucene_index(args.collection, args.index, threads=args.threads)
    print(f"Indexed {len(docs)} documents into {args.index}")


if __name__ == "__main__":
    main()

