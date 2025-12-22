import os

from src.kg.neo4j_client import get_neo4j_database, get_neo4j_driver


def create_property_indexes():
    driver = get_neo4j_driver()
    database = get_neo4j_database()

    queries = [
        "CREATE CONSTRAINT offer_campaign_id IF NOT EXISTS FOR (o:Offer) REQUIRE o.campaignId IS UNIQUE",
        "CREATE INDEX city_name IF NOT EXISTS FOR (c:City) ON (c.name)",
        "CREATE INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)",
        "CREATE INDEX segment_code IF NOT EXISTS FOR (s:SegmentType) ON (s.code)",
        "CREATE INDEX product_code IF NOT EXISTS FOR (p:ProductCode) ON (p.code)",
    ]

    with driver.session(database=database) as session:
        for q in queries:
            session.run(q)

    driver.close()
    print("Property indexes/constraints created.")


def create_vector_index():
    driver = get_neo4j_driver()
    database = get_neo4j_database()

    dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    index_name = os.getenv("NEO4J_VECTOR_INDEX", "offer_embedding_index")

    cypher = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (o:Offer) ON (o.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """

    with driver.session(database=database) as session:
        session.run(cypher)

    driver.close()
    print(f"Vector index ready: {index_name} (dim={dim})")
