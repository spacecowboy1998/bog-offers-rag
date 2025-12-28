from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """
    Structured search plan extracted from the user's natural language query.
    """
    cities: List[str] = Field(
        default_factory=list,
        description="List of specific Georgian cities mentioned (e.g. 'Tbilisi', 'Batumi'). If none, return empty list."
    )
    categories: List[str] = Field(
        default_factory=list,
        description="List of explicit categories (e.g. 'Food', 'Shopping', 'Travel')."
    )
    segment_types: List[str] = Field(
        default_factory=list,
        description="Customer segments (e.g. 'SOLO', 'Student', 'PLUS')."
    )
    product_codes: List[str] = Field(
        default_factory=list,
        description="Specific product codes if mentioned (e.g. 'AMEX', 'VISA')."
    )
    rewritten_queries: List[str] = Field(
        ...,
        description="Generate 3 distinct search queries. 1. Exact product name/type. 2. Semantic intent (e.g. 'romantic dinner'). 3. Broad category.",
        min_length=1
    )

def _load_taxonomy(project_root: Path) -> dict:
    taxonomy_path = project_root / "data" / "processed" / "taxonomy.json"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy.json not found at {taxonomy_path}")

    with taxonomy_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plan_query(client: OpenAI, user_query: str, project_root: Path) -> SearchPlan:
    """
    Uses OpenAI Structured Outputs to convert user query into a validated SearchPlan.
    """
    taxonomy = _load_taxonomy(project_root)

    system_prompt = f"""
    You are an expert query planner for the Bank of Georgia Offers engine.
    
    Your goal is to map the user's input to our internal taxonomy and generate search queries in Georgian.
    
    VALID CITIES: {", ".join(taxonomy.get('cities', []))}...
    VALID CATEGORIES: {", ".join(taxonomy.get('categories', []))}
    VALID SEGMENTS: {", ".join(taxonomy.get('segmentTypes', []))}
    VALID PRODUCT CODES: {", ".join(taxonomy.get('productCodes', []))}
    
    Rules:
    1. EXACT MATCHING: The 'cities', 'categories', 'segment_types', and 'product_codes' fields MUST contain values strictly from the VALID lists provided above.
       - If a user mentions a synonym (e.g. "Eating" instead of "Food"), map it to the closest VALID category.
       - If the user mentions a city NOT in the valid list, ignore it.
    
    2. GEORGIAN QUERIES: The 'rewritten_queries' MUST be in Georgian, even if the user asks in English. The database content is in Georgian.
    
    3. MULTI-CATEGORY STRATEGY: If the user asks for multiple distinct topics (e.g., "I need a hotel and a pharmacy"), you MUST generate at least one specific query for EACH topic.
       Ensure the 'categories' list includes corresponding categories.
       
    4. SEARCH OPTIMIZATION: 'rewritten_queries' should strip filler words and focus on the core product/service.
        - rewritten_queries: must have at least 1 element, If you cannot rewrite, include the original user query.
        - rewritten_queries should be optimized for semantic retrieval. provide maximum 3 queries when possible.
    
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=SearchPlan,
        )

        return completion.choices[0].message.parsed

    except Exception as e:
        print(f"Planner Error: {e}")
        return SearchPlan(rewritten_queries=[user_query])