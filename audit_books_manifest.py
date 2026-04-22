from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='per_book_manifest_106.json')
    ap.add_argument('--documents', default='data/processed/documents.jsonl')
    ap.add_argument('--chunks', default='data/processed/chunks.jsonl')
    ap.add_argument('--output', default='data/processed/per_book_audit_report.json')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    docs = load_jsonl(Path(args.documents))
    chunks = load_jsonl(Path(args.chunks))

    doc_counts = Counter((d.get('author'), d.get('book_title')) for d in docs)
    chunk_counts = Counter((c.get('metadata', {}).get('author'), c.get('metadata', {}).get('book_title')) for c in chunks)
    shortest = defaultdict(lambda: None)
    longest = defaultdict(lambda: None)
    for d in docs:
        key = (d.get('author'), d.get('book_title'))
        text = d.get('text','')
        if shortest[key] is None or len(text) < shortest[key]['chars']:
            shortest[key] = {'title': d.get('title'), 'chars': len(text), 'preview': text[:200]}
        if longest[key] is None or len(text) > longest[key]['chars']:
            longest[key] = {'title': d.get('title'), 'chars': len(text), 'preview': text[:200]}

    report = []
    for item in manifest:
        key = (item['author'], item['file_stem'])
        docs_n = doc_counts.get(key, 0)
        chunks_n = chunk_counts.get(key, 0)
        flags = []
        if item['mode'] == 'single_work' and docs_n != 1:
            flags.append('single_work_doc_count_mismatch')
        if item['mode'] == 'collection' and docs_n < 2:
            flags.append('collection_too_few_docs')
        if docs_n > 1500:
            flags.append('very_high_doc_count')
        if longest[key] and longest[key]['chars'] > 200000 and item['mode'] == 'single_work':
            flags.append('single_work_maybe_frontmatter_or_unsplit')
        if shortest[key] and shortest[key]['chars'] < 120:
            flags.append('contains_very_short_doc')
        row = dict(item)
        row.update({
            'current_docs': docs_n,
            'current_chunks': chunks_n,
            'shortest_doc': shortest[key],
            'longest_doc': longest[key],
            'flags': flags,
        })
        report.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {out} with {len(report)} books; flagged {sum(1 for r in report if r["flags"])} books.')

if __name__ == '__main__':
    main()
