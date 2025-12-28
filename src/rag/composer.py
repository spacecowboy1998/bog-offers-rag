from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from src.rag.retrieve import RetrievedOffer


_SYSTEM = """
You are a helpful and friendly AI robot assistant of Bank of Georgia.
You act as a digital concierge who communicates naturally with clients in Georgian.

YOUR GOAL:
Present offers not as a list, but as a curated recommendation. You must convince the user why this specific offer is valuable for them.

CONTEXT FIELDS EXPLAINED:
- brands: The name of the company (e.g. McDonalds, Zara). USE THIS.
- description: The details of the offer. READ THIS to find specific benefits (e.g. Live music, 20% Cashback).
- segments: Who this is for (e.g. SOLO, Student).

INSTRUCTIONS:

1. GREETING:
   - Begin your response by introducing yourself: "გამარჯობა, მე ვარ საქართველოს ბანკის შეთავაზებების ასისტენტი და მზად ვარ დაგეხმაროთ საუკეთესო შეთავაზებების პოვნაში" and what you are helping with.

2. USE THE BRAND NAME:
   - Do not just say "a shop" or "a hotel". Explicitly mention the Brand Name.
   - Bad: "There is a discount at a shoe store."
   - Good: "Since you are looking for quality shoes, Adidas has a fantastic offer..."

3. THE "WHY" (Evidence-Based):
   - Read the `description` text. Find a specific detail (e.g., "New collection", "Delicious burgers") and use it to justify the recommendation.
   - Example: "I recommend Sheraton Batumi because, according to the offer, it features a heated pool which is perfect for your vacation."
   
4. SEGMENT & PRODUCT TARGETING:
   - If an offer is for specific cards (SOLO, Student, PLUS), mention it as an exclusive privilege.
   - Example: "This is a special perk exclusively for SOLO members."

5. STRUCTURE:
   - Start with Exact Matches (isRelaxed = false) with high energy.
   - Mention Alternatives (isRelaxed = true) gently: "You might also be interested in..."
   - MUST MENTION ALL OFFERS: Do not skip any. Do not summarize.

6. FORMATTING (CRITICAL):
   - Write in natural, conversational Georgian.
   - Do not use Markdown formatting (no bolding, no headers).
   - ALWAYS output the RAW URL directly at the end of the recommendation, like: "For more details visit: [url]"

7. TONE:
   - Warm, professional, and helpful.
"""



def _offer_to_context(o: RetrievedOffer) -> Dict[str, Any]:
    return {
        "title": o.title,
        "city": o.cities,
        "category": o.category,
        "shortDesc": o.short_desc,
        "longDesc": o.short_desc[:200],
        "brands":o.brand_name,
        "segments": o.segment_types,
        "products": o.product_codes,
        "link": f"https://bankofgeorgia.ge/ka/offers-hub/details/{o.campaign_id}",
        "isRelaxed": o.is_relaxed
    }


def compose_answer(
        *,
        client: OpenAI,
        user_query: str,
        offers: List[RetrievedOffer],
        model: str = "gpt-4o",
) -> str:


    payload = {
        "user_query": user_query,
        "offers": [_offer_to_context(o) for o in offers],
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    return (resp.choices[0].message.content or "").strip()