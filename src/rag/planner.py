from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


_SYSTEM_PROMPT = """
You are an assistant for the Bank of Georgia Offers Hub.

Task: Convert the user's natural-language query into a Retrieval Plan.
Return ONLY valid JSON. No markdown, no explanations.

Rules:
- Output must follow the schema exactly.
- city: must be either an exact item from known_cities OR null.
- categories: select ONLY from allowed_categories.
- segmentTypes: select ONLY from allowed_segmentTypes.
- productCodes: select ONLY from allowed_productCodes.
- If not clearly implied, leave the list empty.
- rewritten_queries: must have at least 1 element.
  - If you cannot rewrite, include the original user query.
- rewritten_queries should be optimized for semantic retrieval:
  - keep them short
  - include city/category keywords if helpful
  - provide 2–4 queries when possible

Schema:
{
  "city": string | null,
  "categories": [string],
  "segmentTypes": [string],
  "productCodes": [string],
  "rewritten_queries": [string]
}
"""


def _load_taxonomy(project_root: Path) -> Dict[str, Any]:
    taxonomy_path = project_root / "data" / "processed" / "taxonomy.json"
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"taxonomy.json not found: {taxonomy_path}. "
            "Generate it in fetch_offers.py first."
        )

    with taxonomy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("taxonomy.json must be a JSON object")
    return data


def plan_query(client: OpenAI, user_query: str, project_root: Path) -> Dict[str, Any]:
    taxonomy = _load_taxonomy(project_root)

    known_cities = taxonomy.get("cities", [])
    allowed_categories = taxonomy.get("categories", [])
    allowed_segment_types = taxonomy.get("segmentTypes", [])
    allowed_product_codes = taxonomy.get("productCodes", [])

    user_payload = {
        "user_query": user_query,
        "known_cities": known_cities,
        "allowed_categories": allowed_categories,
        "allowed_segmentTypes": allowed_segment_types,
        "allowed_productCodes": allowed_product_codes,
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {content}") from e

    # Defaults
    plan.setdefault("city", None)
    plan.setdefault("categories", [])
    plan.setdefault("segmentTypes", [])
    plan.setdefault("productCodes", [])
    plan.setdefault("rewritten_queries", [])

    if not plan["rewritten_queries"]:
        plan["rewritten_queries"] = [user_query]

    # Hard-safety
    if plan["city"] not in known_cities:
        plan["city"] = None

    if not isinstance(plan["categories"], list):
        plan["categories"] = []
    if not isinstance(plan["segmentTypes"], list):
        plan["segmentTypes"] = []
    if not isinstance(plan["productCodes"], list):
        plan["productCodes"] = []
    if not isinstance(plan["rewritten_queries"], list) or not plan["rewritten_queries"]:
        plan["rewritten_queries"] = [user_query]

    plan["categories"] = [c for c in plan["categories"] if c in allowed_categories]
    plan["segmentTypes"] = [s for s in plan["segmentTypes"] if s in allowed_segment_types]
    plan["productCodes"] = [p for p in plan["productCodes"] if p in allowed_product_codes]

    return plan
