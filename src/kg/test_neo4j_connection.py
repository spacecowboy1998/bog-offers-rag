from src.kg.neo4j_client import get_neo4j_driver, get_neo4j_database


def main():
    driver = get_neo4j_driver()
    database = get_neo4j_database()

    with driver.session(database=database) as session:
        result = session.run("RETURN 'Neo4j connected' AS msg")
        print(result.single()["msg"])

    driver.close()

if __name__ == "__main__":
    main()
