from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.config import settings
from rag_system.utils.io import read_jsonl, write_jsonl_dicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple evaluation template from built chunks.")
    parser.add_argument("--output", default="evaluation_template.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    chunks = read_jsonl(settings.chunks_path)
    rows = []
    for chunk in chunks[: args.limit]:
        rows.append(
            {
                "question": f"{chunk['title']} sənədində hansı əsas məlumat verilir?",
                "gold_answer": chunk["text"][:300],
                "gold_doc_ids": [chunk["doc_id"]],
            }
        )
    output_path = settings.evaluation_dir / Path(args.output).name
    write_jsonl_dicts(output_path, rows)
    print(f"Template written to: {output_path}")


if __name__ == "__main__":
    main()
