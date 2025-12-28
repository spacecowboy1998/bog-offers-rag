# Bank of Georgia Offers GraphRAG

**An intelligent Graph-based Retrieval-Augmented Generation (RAG) engine that integrates a Neo4j Property Graph with Vector Embeddings to provide high-precision financial offer discovery.**

## Project Status

The core engine is fully functional. The system currently supports automated ETL from the Bank of Georgia API, Neo4j graph construction with vector indexing, and an agentic RAG search pipeline.

## Project Screenshots

## Installation and Setup Instructions

**Prerequisites:** You will need Python 3.10+, a Neo4j instance (local or AuraDB), and an OpenAI API Key.

### Clone the Repository

```bash
git clone https://github.com/yourusername/bog-offers-rag.git
cd bog-offers-rag

```

### Setup Steps

1. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


2. **Configure Environment:**
Create a `.env` file in the root directory with the following keys:
```ini
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o-mini

```


3. **Ingest Data & Build Graph:**
```bash
python -m scripts.fetch_offers
python -m scripts.neo4j_ops

```


4. **Start Search CLI:**
```bash
python -m scripts.search_cli

```



---

## Reflection

### Context

This project was built as a high-fidelity implementation of **Hybrid GraphRAG** designed to handle complex relational data within a banking ecosystem. The goal was to move beyond basic vector search and incorporate structured business logic—such as cities, brands, and eligibility segments—directly into the retrieval process.

### The Build

I set out to build an engine that could intelligently handle the "Zero Result" problem. By utilizing **Soft Filtering** and **Weighted Scoring**, the system ensures that if an exact match (e.g., "Sushi in Batumi") doesn't exist, it can provide semantically similar alternatives while explicitly labeling them as recommendations rather than search failures.

 $Score = \text{VectorScore} + (2.5 \times \text{CityMatch}) + (2.0 \times \text{BrandMatch}) + (1.5 \times \text{CategoryMatch})$
---

## Project Structure

```text
bog-offers-rag/
├── data/                   # JSON storage for raw and processed offer data
├── scripts/                # Execution entry points
│   ├── fetch_offers.py     # ETL: Fetch from API -> Process -> Save JSON
│   ├── neo4j_ops.py        # ETL: Ingest JSON -> Neo4j Graph + Embeddings
│   └── search_cli.py       # RAG: Interactive CLI for testing queries
├── src/
│   ├── data_collection/    # API Client and Pydantic validation models
│   ├── kg/                 # Graph Builders, Batch Ingestion, and Indexing
│   ├── rag/                # RAG Logic (Planner, Weighted Retrieve, Composer)
│   └── Utils/              # Environment and credential helpers
├── .env                    # Local environment secrets
├── .gitignore              # Project-specific git exclusion rules
└── requirements.txt        # Pinned project dependencies

```
