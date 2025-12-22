from src.kg.neo4j_client import get_neo4j_database, get_neo4j_driver


def print_graph_stats():
    driver = get_neo4j_driver()
    database = get_neo4j_database()

    q = """
    MATCH (o:Offer) WITH count(o) AS offers
    MATCH (c:City) WITH offers, count(c) AS cities
    MATCH (cat:Category) WITH offers, cities, count(cat) AS categories
    RETURN offers, cities, categories
    """

    with driver.session(database=database) as session:
        row = session.run(q).single()
        print(dict(row))

    driver.close()
