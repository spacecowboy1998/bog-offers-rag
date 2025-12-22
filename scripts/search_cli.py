import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from Utils.helper import get_openai_api_key

from src.kg.neo4j_client import get_neo4j_database, get_neo4j_driver
from src.rag.retrieve import retrieve_candidates_graph
from src.rag.merge import merge_candidates
from src.rag.planner import plan_query
from src.rag.composer import compose_answer


def main():
    try:
        user_query = input("Please enter your query: ").strip()
    except KeyboardInterrupt:
        sys.exit(0)

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")

    project_root = Path(__file__).resolve().parents[1]

    openai_client = OpenAI(api_key=api_key)
    driver = get_neo4j_driver()
    database = get_neo4j_database()

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    index_name = os.getenv("NEO4J_VECTOR_INDEX", "offer_embedding_index")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o")

    #  PLAN
    plan: Dict[str, Any] = plan_query(openai_client, user_query, project_root)

    #  RETRIEVE
    all_hits: List[Any] = []

    for rq in plan.get("rewritten_queries", []):
        hits = retrieve_candidates_graph(
            driver=driver,
            database=database,
            openai_client=openai_client,
            embedding_model=embedding_model,
            index_name=index_name,
            query=rq,
            top_k=12,
            city=plan.get("city"),
            categories=plan.get("categories", []),
            segment_types=plan.get("segmentTypes", []),
            product_codes=plan.get("productCodes", []),
        )
        all_hits.extend(hits)

    # MERGE & RANK
    merged = merge_candidates(all_hits)
    final_offers = merged[:7]

    #  GENERATE
    answer = compose_answer(
        client=openai_client,
        user_query=user_query,
        plan=plan,
        offers=final_offers,
        model=chat_model,
    )

    print(answer)

    driver.close()


if __name__ == "__main__":
    main()