
# Bank of Georgia Offers GraphRAG

**An intelligent Schema-Driven Graph RAG engine that combines Neo4j Knowledge Graphs with Vector Embeddings to provide high-precision, compliant financial offer recommendations.**

Unlike standard RAG, which relies solely on semantic similarity ("vibe matching"), this system uses a **Graph-Guarded Retrieval** strategy. It enforces strict business logic (Location, Expiry, Eligibility) while leveraging LLMs for natural language understanding and persuasive communication.

---

## 🚀 How It Works (The 3-Step Pipeline)

The system transforms unstructured user queries into structured graph operations through a three-stage pipeline:

### **Step 1: Intelligent Data Ingestion (ETL)**
We do not scrape HTML. We reverse-engineered the internal API to fetch raw JSON data, ensuring reliability.
* **Action:** Fetches offers from `bankofgeorgia.ge`, cleans HTML tags, parses dates, and normalizes metadata.
* **Graph Construction:** Creates a Neo4j Knowledge Graph where `Offers` are nodes connected to `City`, `Category`, `Brand`, and `Segment` entities.
* **Vectorization:** Generates semantic embeddings using `text-embedding-3-large` on a rich text block (Title + Brand + Long Description).

### **Step 2: Schema-Driven Intent Planning**
We replaced fragile "prompt engineering" with **Pydantic Structured Outputs**.
* **Action:** When a user asks *"I need a hotel in Batumi for a food tour,"* the Planner forces the LLM to output a strict Python object:
    ```python
    SearchPlan(
        cities=["Batumi"],
        categories=["Food", "Travel"],
        rewritten_queries=["Best restaurants Batumi", "Hotels in Batumi"]
    )
    ```
* **Benefit:** This guarantees type safety and eliminates hallucinations (e.g., inventing cities that don't exist in our database).

### **Step 3: Graph-Guarded Hybrid Retrieval**
We execute a **Hybrid Search** that combines the speed of vectors with the accuracy of graphs.
* **Vector Search:** Finds the top 100 offers semantically related to "food tour".
* **Graph Filter (The Guard):** Applies strict Cypher constraints:
    * `WHERE o.endDate >= date()` (No expired offers)
    * `MATCH (o)-[:IN_CITY]->({name: 'Batumi'})` (Strict location filtering)
* **Result:** The system physically excludes irrelevant offers (e.g., a sushi place in Tbilisi) regardless of how high their vector score is.

---

## 🛠️ Project Structure

```text
bog-offers-rag/
├── data/                   # Local storage for raw/processed JSON
├── scripts/                # Execution entry points
│   ├── fetch_offers.py     # Step 1: API Extraction & Cleaning
│   ├── neo4j_ops.py        # Step 1: Graph Building & Indexing
│   └── search_cli.py       # The Main Application (Planner -> Retrieve -> Chat)
├── src/
│   ├── data_collection/    # Pydantic models for data validation
│   ├── kg/                 # Graph Builders, Batch Ingestion, and Indexing
│   ├── rag/                # The Brain:
│   │   ├── planner.py      # Pydantic Router (User Intent -> Schema)
│   │   ├── retrieve.py     # Graph-Guarded Search (Cypher + Vectors)
│   │   ├── merge.py        # Deduplication & Re-ranking Logic
│   │   └── composer.py     # Consultant Agent (Generates "Why" explanations)
└── .env                    # Secrets configuration

```

---

## 💻 Installation & Setup

**Prerequisites:** Python 3.10+, Neo4j (Local or AuraDB), OpenAI API Key.

### 1. Clone & Install

```bash
git clone [https://github.com/yourusername/bog-offers-rag.git](https://github.com/yourusername/bog-offers-rag.git)
cd bog-offers-rag
pip install -r requirements.txt

```

### 2. Configure Environment

Create a `.env` file in the root directory:

```ini
OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o

```

### 3. Build the Brain (Run Once)

This fetches live data, processes it, and constructs the Knowledge Graph.

```bash
# Fetch fresh data from Bank of Georgia
python -m scripts.fetch_offers

# Build Graph & Vector Indexes
python -m scripts.neo4j_ops

```

### 4. Run the Concierge

Start the interactive chat session.

```bash
python -m scripts.search_cli
