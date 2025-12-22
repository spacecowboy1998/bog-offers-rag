from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Driver
from openai import OpenAI


@dataclass(frozen=True)
class RetrievedOffer:
    campaign_id: int
    title: str
    category: str
    short_desc: str
    cities: str
    brands: str
    segment_types: str
    product_codes: str
    score: float
    matched_query: str
    is_relaxed: bool


def _format_list_prop(val: Any) -> str:
    """Helper to safely convert Neo4j lists to a string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val).strip()


def retrieve_candidates_graph(
        *,
        driver: Driver,
        database: str,
        openai_client: OpenAI,
        embedding_model: str,
        index_name: str,
        query: str,
        top_k: int,
        city: Optional[str],
        categories: List[str],
        segment_types: List[str],
        product_codes: List[str],
) -> List[RetrievedOffer]:

    # 1. Embed the query
    emb = openai_client.embeddings.create(model=embedding_model, input=[query])
    vector = emb.data[0].embedding

    # 2. Smart Weighted Cypher Query
    cypher = """
    CALL db.index.vector.queryNodes($indexName, 50, $vector)
    YIELD node AS o, score AS vectorScore

    // Check City Match (Weight: 2.5)
    OPTIONAL MATCH (o)-[:IN_CITY]->(c:City)
    WITH o, vectorScore, c,
         CASE WHEN ($city IS NOT NULL AND toLower(c.name) CONTAINS toLower($city)) THEN 1 ELSE 0 END AS cityMatch

    // Check Category Match (Weight: 1.5)
    OPTIONAL MATCH (o)-[:IN_CATEGORY]->(cat:Category)
    WITH o, vectorScore, c, cityMatch, cat,
         CASE WHEN ($categories IS NOT NULL AND cat.name IN $categories) THEN 1 ELSE 0 END AS catMatch

    // Check Brand Match (Weight: 2.0)
    OPTIONAL MATCH (o)-[:HAS_BRAND]->(b:Brand)
    WITH o, vectorScore, c, cityMatch, cat, catMatch, b,
         CASE WHEN toLower($query) CONTAINS toLower(b.name) THEN 1 ELSE 0 END AS brandMatch

    // Check Segment Match (Weight: 1.0)
    OPTIONAL MATCH (o)-[:REQUIRES_SEGMENT]->(s:SegmentType)
    WITH o, vectorScore, c, cityMatch, cat, catMatch, b, brandMatch, s,
         CASE WHEN ($segmentTypes IS NOT NULL AND s.code IN $segmentTypes) THEN 1 ELSE 0 END AS segMatch


    WITH o, vectorScore, c, cat, b, s, cityMatch, catMatch, brandMatch, segMatch,
         (vectorScore 
          + (cityMatch * 2.5) 
          + (brandMatch * 2.0) 
          + (catMatch * 1.5) 
          + (segMatch * 1.0)) AS finalScore

    // Aggregation (Deduplicate rows)
    WITH o, max(finalScore) AS score, 
         max(cityMatch) AS cityHit,      
         max(catMatch) AS catHit,        
         collect(distinct c.name) AS relCities

    // It is "Relaxed" (Alternative) ONLY if a specific requested constraint FAILED.
    WITH o, score, relCities,
         CASE 
           // User asked for City, but we found 0 matches for it
           WHEN ($city IS NOT NULL AND cityHit = 0) THEN true
           
           // User asked for Category, but we found 0 matches for it
           WHEN ($categories IS NOT NULL AND size($categories) > 0 AND catHit = 0) THEN true
           
           // Otherwise, it is a valid match
           ELSE false 
         END AS isRelaxed

    RETURN
      o.campaignId AS campaignId,
      o.title AS title,
      o.categoryDesc AS categoryDesc,
      o.shortDesc AS shortDesc,
      o.cityNames AS cityNames,
      o.brandNames AS brandNames,
      o.segmentTypesFlat AS segmentTypesFlat,
      o.productCodesFlat AS productCodesFlat,
      score,
      isRelaxed

    ORDER BY score DESC
    LIMIT $topK
    """

    params = {
        "indexName": index_name,
        "vector": vector,
        "topK": top_k,
        "query": query,
        "city": city,
        "categories": categories,
        "segmentTypes": segment_types,
        "productCodes": product_codes
    }

    out: List[RetrievedOffer] = []

    with driver.session(database=database) as session:
        res = session.run(cypher, params)
        for r in res:
            out.append(
                RetrievedOffer(
                    campaign_id=int(r["campaignId"]),
                    title=str(r["title"] or "").strip(),
                    category=str(r["categoryDesc"] or "").strip(),
                    short_desc=str(r["shortDesc"] or "").strip(),
                    cities=_format_list_prop(r["cityNames"]),
                    brands=_format_list_prop(r["brandNames"]),
                    segment_types=_format_list_prop(r["segmentTypesFlat"]),
                    product_codes=_format_list_prop(r["productCodesFlat"]),
                    score=float(r["score"]),
                    matched_query=query,
                    is_relaxed=bool(r["isRelaxed"])
                )
            )

    return out