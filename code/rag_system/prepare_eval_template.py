from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.config import settings
from rag_system.utils.io import read_jsonl, write_jsonl_dicts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an evaluation template from built chunks. "
        "Fill gold_answer manually before running evaluation."
    )
    parser.add_argument("--output", default="evaluation_template.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--autofill-gold-from-reference",
        action="store_true",
        help="Fill gold_answer with a short excerpt from the reference chunk. Useful for quick smoke tests only.",
    )
    args = parser.parse_args()

    chunks = read_jsonl(settings.chunks_path)
    rows = []
    seen_doc_ids: set[str] = set()

    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        reference_text = str(chunk.get("text", "")).strip()
        title = str(chunk.get("title", "")).strip() or doc_id
        section_title = str(chunk.get("section_title", "")).strip()
        source = str(chunk.get("source", "")).strip()

        question = f"'{title}' mövzusu haqqında əsas məlumat nədir?"
        if section_title:
            question = f"'{title}' sənədinin '{section_title}' bölməsində hansı əsas məlumat verilir?"

        row = {
            "question": question,
            "gold_answer": reference_text[:300] if args.autofill_gold_from_reference else "",
            "gold_doc_ids": [doc_id],
            "gold_chunk_ids": [str(chunk.get("chunk_id", ""))],
            "metadata": {
                "reference_title": title,
                "reference_section_title": section_title,
                "reference_source": source,
                "reference_chunk_id": str(chunk.get("chunk_id", "")),
                "reference_text": reference_text[:1200],
                "needs_manual_gold_answer": not args.autofill_gold_from_reference,
            },
        }
        rows.append(row)
        if len(rows) >= args.limit:
            break

    output_path = settings.evaluation_dir / Path(args.output).name
    write_jsonl_dicts(output_path, rows)
    print(f"Template written to: {output_path}")


if __name__ == "__main__":
    main()
