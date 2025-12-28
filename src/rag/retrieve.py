from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from neo4j import Driver
from openai import OpenAI


@dataclass(frozen=True)
class RetrievedOffer:
    campaign_id: int
    title: str
    category: str
    short_desc: str
    cities: str
    long_desc: str
    brand_name: str
    segment_types: str
    product_codes: str
    score: float
    is_relaxed: bool


def retrieve_candidates_graph(
        *,
        driver: Driver,
        database: str,
        openai_client: OpenAI,
        embedding_model: str,
        index_name: str,
        query: str,
        top_k: int = 15,
        cities: List[str] = None,
        categories: List[str] = None,
        segment_types: List[str] = None,
        product_codes: List[str] = None,
) -> List[RetrievedOffer]:

    # 1. Generate Vector
    emb = openai_client.embeddings.create(model=embedding_model, input=[query])
    vector = emb.data[0].embedding

    # 2. Graph-Guarded Cypher Query
    cypher = """
    // A. Vector Search
    CALL db.index.vector.queryNodes($indexName, 100, $vector)
    YIELD node AS o, score AS vectorScore


    // Conditional Hard Filter: City
    MATCH (o)-[:IN_CITY]->(c:City)
    WITH o, vectorScore, collect(c.name) as offerCities
    WHERE 
        size($cities) = 0 
        OR 
        any(city IN $cities WHERE city IN offerCities)

    // Conditional Hard Filter: Segments
    OPTIONAL MATCH (o)-[:REQUIRES_SEGMENT]->(s:SegmentType)
    WITH o, vectorScore, offerCities, collect(s.code) as segments
    WHERE 
        size($segmentTypes) = 0 
        OR 
        any(seg IN segments WHERE seg IN $segmentTypes)
    
    // Conditional Hard Filter: Products (UPDATED)
    OPTIONAL MATCH (o)-[:REQUIRES_PRODUCT]->(p:ProductCode)
    WITH o, vectorScore, offerCities, segments, collect(p.code) as products
    WHERE 
        size($productCodes) = 0 
        OR 
        any(prod IN products WHERE prod IN $productCodes)

    //  Fetch Products (For context/display)
    OPTIONAL MATCH (o)-[:REQUIRES_PRODUCT]->(p:ProductCode)
    WITH o, vectorScore, offerCities, segments, collect(p.code) as products

    //  Soft Filter: Category (Boost Score)
    OPTIONAL MATCH (o)-[:IN_CATEGORY]->(cat:Category)
    WITH o, vectorScore, offerCities, segments, products, cat,
         CASE 
            WHEN size($categories) > 0 AND cat.name IN $categories THEN 1.0 
            ELSE 0.0 
         END AS catBoost
    
    // Get Brand for context
    OPTIONAL MATCH (o)-[:HAS_BRAND]->(b:Brand)

    //Final Scoring
    WITH o, offerCities, segments, products, b, cat, 
         (vectorScore + (catBoost * 0.1)) as finalScore,
         (size($categories) > 0 AND catBoost = 0) as isRelaxed

    RETURN
      o.campaignId AS campaignId,
      o.title AS title,
      o.categoryDesc AS categoryDesc,
      o.shortDesc AS shortDesc,
      o.longDesc AS longDesc,
      offerCities AS cityNames,
      segments AS segmentTypes,
      products AS productCodes,
      b.name AS brandName,
      finalScore AS score,
      isRelaxed

    ORDER BY finalScore DESC
    LIMIT toInteger($topK)
    """

    params = {
        "indexName": index_name,
        "vector": vector,
        "topK": top_k,
        "cities": cities or [],
        "categories": categories or [],
        "segmentTypes": segment_types or [],
        "productCodes": product_codes or []
    }

    results: List[RetrievedOffer] = []

    with driver.session(database=database) as session:
        rows = session.run(cypher, params)
        for r in rows:
            # Safe string formatting for lists
            city_str = ", ".join(r["cityNames"]) if r["cityNames"] else "All Cities"
            seg_str = ", ".join(r["segmentTypes"]) if r["segmentTypes"] else ""
            prod_str = ", ".join(r["productCodes"]) if r["productCodes"] else ""

            results.append(RetrievedOffer(
                campaign_id=r["campaignId"],
                title=r["title"],
                category=r["categoryDesc"],
                short_desc=r["shortDesc"],
                long_desc=r["longDesc"],
                cities=city_str,
                brand_name=r["brandName"] or "",
                segment_types=seg_str,
                product_codes=prod_str,
                score=r["score"],
                is_relaxed=r["isRelaxed"]
            ))

    return results