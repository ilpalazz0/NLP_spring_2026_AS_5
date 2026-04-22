from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    return float(any(item_id in gold for item_id in retrieved_ids))


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    for idx, item_id in enumerate(retrieved_ids, start=1):
        if item_id in gold:
            return 1.0 / idx
    return 0.0
