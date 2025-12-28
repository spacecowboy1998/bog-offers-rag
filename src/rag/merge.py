from __future__ import annotations

from typing import Dict, List

from src.rag.retrieve import RetrievedOffer


def merge_candidates(candidates: List[RetrievedOffer]) -> List[RetrievedOffer]:
    """
    Merge candidates across multiple rewritten queries.
    Keep the best-scoring hit per campaign_id.

    Prioritizes Exact Matches (is_relaxed=False) over Relaxed ones.
    """
    best: Dict[int, RetrievedOffer] = {}

    for c in candidates:
        existing = best.get(c.campaign_id)

        if existing is None:
            best[c.campaign_id] = c
        else:
            if not c.is_relaxed and existing.is_relaxed:
                best[c.campaign_id] = c
            elif c.is_relaxed == existing.is_relaxed:
                if c.score > existing.score:
                    best[c.campaign_id] = c

    merged = list(best.values())

    merged.sort(key=lambda x: (x.is_relaxed, -x.score))

    return merged