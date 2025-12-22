import os
from Utils.helper import load_env
from neo4j import GraphDatabase, Driver


def get_neo4j_driver() -> Driver:
    """
    Create and return a Neo4j driver instance.
    """
    load_env()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise RuntimeError("Missing Neo4j connection environment variables.")

    return GraphDatabase.driver(uri, auth=(user, password))


def get_neo4j_database() -> str:
    """
    Returns target Neo4j database name.
    """
    return os.getenv("NEO4J_DATABASE", "neo4j")
