import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from Utils.helper import get_openai_api_key
from src.kg.neo4j_client import get_neo4j_database, get_neo4j_driver


def _as_list(value: Any):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    return [str(value).strip()]


def _build_doc_text(offer: Dict[str, Any]):
    """
    Builds semantically dense text for embedding.
    Combines Title + Brand + Short Desc + Long Desc.
    """
    title = offer.get("title", "") or ""

    # Brands are critical keywords
    brands = offer.get("brandNames", [])
    if isinstance(brands, str):
        brands = [brands]

    # Clean and retrieve descriptions
    short_desc = (offer.get("shortDesc") or "").strip()
    long_desc = (offer.get("longDesc") or "").strip()

    full_description = f"{short_desc}\n{long_desc}"

    return (
        f"Offer: {title}\n"
        f"Brand: {', '.join(brands)}\n"
        f"Details: {full_description[:1000]}"
    ).strip()


def _load_processed_offers(project_root) :
    stamp = date.today().isoformat()
    path = project_root / "data" / "processed" / f"offers_{stamp}.json"
    if not path.exists():
        # Fallback to try finding ANY file if today's doesn't exist (optional)
        raise FileNotFoundError(f"Processed offers not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Processed offers file must contain a list.")
    return data


def build_graph_with_embeddings():
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))

    project_root = Path(__file__).resolve().parents[2]
    offers = _load_processed_offers(project_root)

    # Prepare embedding inputs
    items: List[Dict[str, Any]] = []
    for offer in offers:
        cid = offer.get("campaignId")
        if cid is None:
            continue
        items.append(
            {
                "campaignId": int(cid),
                "doc": _build_doc_text(offer),
                "offer": offer,
            }
        )

    driver = get_neo4j_driver()
    database = get_neo4j_database()

    cypher = """
    MERGE (o:Offer {campaignId: $campaignId})
    SET
      o.title = $title,
      o.categoryDesc = $categoryDesc,
      o.shortDesc = $shortDesc,
      o.longDesc = $longDesc,
      o.benefText = $benefText,
      o.startDate = $startDate, 
      o.endDate = $endDate,      
      o.cityNames = $cityNamesFlat,
      o.brandNames = $brandNamesFlat,
      o.segmentTypesFlat = $segmentTypesFlat,
      o.productCodesFlat = $productCodesFlat,
      
      o.embedding = $embedding

    WITH o

    // Iterate over LISTS to create relationships
    FOREACH (city IN $cityNamesList |
      MERGE (c:City {name: city})
      MERGE (o)-[:IN_CITY]->(c)
    )

    FOREACH (cat IN $categoriesList |
      MERGE (k:Category {name: cat})
      MERGE (o)-[:IN_CATEGORY]->(k)
    )

    FOREACH (b IN $brandNamesList |
      MERGE (br:Brand {name: b})
      MERGE (o)-[:HAS_BRAND]->(br)
    )

    FOREACH (st IN $segmentTypesList |
      MERGE (s:SegmentType {code: st})
      MERGE (o)-[:REQUIRES_SEGMENT]->(s)
    )

    FOREACH (pc IN $productCodesList |
      MERGE (p:ProductCode {code: pc})
      MERGE (o)-[:REQUIRES_PRODUCT]->(p)
    )
    """

    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            docs = [x["doc"] for x in chunk]

            emb = client.embeddings.create(model=embedding_model, input=docs)
            vectors = [row.embedding for row in emb.data]

            tx = session.begin_transaction()
            try:
                for item, vec in zip(chunk, vectors):
                    offer = item["offer"]

                    city_list = _as_list(offer.get("cityNames"))
                    brand_list = _as_list(offer.get("brandNames"))
                    segment_list = _as_list(offer.get("segmentTypes"))
                    product_list = _as_list(offer.get("productCodes"))

                    category_desc = (offer.get("categoryDesc") or "").strip()
                    category_list = [category_desc] if category_desc else []

                    tx.run(
                        cypher,
                        {
                            "campaignId": item["campaignId"],
                            "title": offer.get("title") or "",
                            "categoryDesc": category_desc,
                            "shortDesc": offer.get("shortDesc") or "",
                            "longDesc": (offer.get("longDesc") or "")[:600],
                            "benefText": offer.get("benefText") or "",
                            "startDate": offer.get("startDate"),
                            "endDate": offer.get("endDate"),
                            "segmentTypesFlat": ", ".join(segment_list),
                            "productCodesFlat": ", ".join(product_list),
                            "cityNamesFlat": ", ".join(city_list),
                            "brandNamesFlat": ", ".join(brand_list),
                            "embedding": vec,

                            # Lists for relationships
                            "cityNamesList": city_list,
                            "brandNamesList": brand_list,
                            "segmentTypesList": segment_list,
                            "productCodesList": product_list,
                            "categoriesList": category_list,
                        },
                    )

                    total += 1

                tx.commit()
            except Exception:
                tx.rollback()
                raise

            print(f"Upserted and Embedded: {total}/{len(items)}")

    driver.close()
    print("Graph build (with embeddings) complete.")