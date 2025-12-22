from __future__ import annotations

from typing import Dict, List

from src.rag.retrieve import RetrievedOffer


def merge_candidates(candidates: List[RetrievedOffer]) -> List[RetrievedOffer]:
    """
    Merge candidates across multiple rewritten queries.
    Keep the best-scoring hit per campaign_id.
    """
    best: Dict[int, RetrievedOffer] = {}

    for c in candidates:
        existing = best.get(c.campaign_id)
        if existing is None or c.score > existing.score:
            best[c.campaign_id] = c

    merged = list(best.values())
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged
