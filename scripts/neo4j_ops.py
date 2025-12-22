from src.kg.indexes import create_property_indexes, create_vector_index
from src.kg.graph_build import build_graph_with_embeddings
from src.kg.graph_stats import print_graph_stats


def main():

    # property indexes (fast lookups / merges)
    create_property_indexes()

    # vector index (needed for similarity search)
    create_vector_index()

    # build graph (nodes + relationships + embeddings)
    build_graph_with_embeddings()

    # sanity stats
    print_graph_stats()


if __name__ == "__main__":
    main()
