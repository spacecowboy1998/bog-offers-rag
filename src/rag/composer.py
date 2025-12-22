from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from src.rag.retrieve import RetrievedOffer


_SYSTEM = """
You are a helpful AI Concierge for Bank of Georgia.
Your task is to have a natural conversation with a client in Georgian and suggest banking offers.

Context:
You will receive a list of offers.
- isRelaxed = false: These are Exact Matches.
- isRelaxed = true: These are Recommendations (Alternatives).

Instructions:
1. Start with the Exact Matches (if any). Be enthusiastic and direct.
2. Only mention Alternative offers if they exist in the list. If the list contains only exact matches, do not mention alternatives at all.
3. Do not use Markdown formatting like bolding (**) or headers (##). Write in clean, plain text.
4. Avoid rigid listings like "City: Batumi, Category: Food". Instead, weave the details into a natural summary sentence.
   - Example: "I recommend visiting Sheraton Batumi for your vacation, which offers a great discount."
5. If an offer mentions specific Segment Types (e.g., SOLO, Student, PLUS) or Product Codes, you MUST mention them naturally in the sentence (e.g., "This offer is exclusively for SOLO cardholders" or "Available for Student card users").
6. For every offer, you must provide the link. Phrase it naturally, like: "For more details visit: [url]"
7. Keep it concise. Two or three sentences per offer is enough.

Goal:
Sound like a helpful consultant chatting with a client, not a robot reading a database.
"""


def _offer_to_context(o: RetrievedOffer) -> Dict[str, Any]:
    return {
        "title": o.title,
        "city": o.cities,
        "category": o.category,
        "longDesc": o.short_desc[:150],
        "segments": o.segment_types,
        "products": o.product_codes,
        "link": f"https://bankofgeorgia.ge/ka/offers-hub/details/{o.campaign_id}",
        "isRelaxed": o.is_relaxed
    }


def compose_answer(
        *,
        client: OpenAI,
        user_query: str,
        plan: Dict[str, Any],
        offers: List[RetrievedOffer],
        model: str = "gpt-4o",
) -> str:
    # Sort: Strict matches first, then Relaxed matches
    sorted_offers = sorted(offers, key=lambda x: x.is_relaxed)

    payload = {
        "user_query": user_query,
        "offers": [_offer_to_context(o) for o in sorted_offers],
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    return (resp.choices[0].message.content or "").strip()