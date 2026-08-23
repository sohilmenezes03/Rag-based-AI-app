 Python RAG Application

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about local .txt and .pdf documents.

The application combines keyword-based retrieval, semantic search, ensemble retrieval, cross-encoder reranking, and Google Gemini to generate answers grounded in the provided documents.

 ## Architecture

```text
                    ┌─────────────────┐
                    │  Local Documents │
                    │  TXT / PDF Files │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Document Loading │
                    │   + Chunking     │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌───────────────┐         ┌───────────────┐
        │     BM25      │         │     FAISS     │
        │ Keyword Search│         │ Semantic Search│
        └───────┬───────┘         └───────┬───────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Weighted Ensemble│
                    │    Retrieval     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Cross-Encoder   │
                    │    Reranker     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevance Filter│
                    │ + Source Context│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Google Gemini   │
                    │ Answer Generation│
                    └────────┬────────┘
                             │
                             ▼
                    Answer + Source Chunks
```
Features
- Supports .txt and .pdf documents.
- Splits documents into overlapping chunks.
- Uses BM25 for keyword-based retrieval.
- Uses FAISS and Sentence Transformers for semantic similarity search.
- Combines BM25 and FAISS using a weighted ensemble.
- Uses a Cross-Encoder reranker to improve the ranking of retrieved chunks.
- Filters weakly relevant results.
- Preserves useful context from the same source document.
- Generates grounded answers using Google Gemini.
- Displays retrieved source chunks and reranker scores.
- Includes a small evaluation script for testing retrieval quality.

Retrieval Pipeline

The application uses a multi-stage retrieval pipeline:

User Query
    ↓
BM25 Retrieval
    +
FAISS Semantic Retrieval
    ↓
Weighted Ensemble
    ↓
Top Candidate Chunks
    ↓
Cross-Encoder Reranking
    ↓
Relevance Filtering
    ↓
Same-Document Context Preservation
    ↓
Relevant Context
    ↓
Google Gemini
    ↓
Final Answer with Source Attribution
Why Multiple Retrievers?

BM25 is useful for exact keyword matching.

FAISS performs semantic search using vector embeddings, allowing the system to find documents with similar meaning even when the exact words differ.

The two retrieval methods are combined using a weighted ensemble:

BM25 Weight: 0.3
FAISS Weight: 0.7
Reranking

After the ensemble retrieval stage, the candidate chunks are passed to a Cross-Encoder:

cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker evaluates the relationship between the user query and each retrieved chunk and produces a more refined ranking.

The system also applies relevance filtering while allowing useful additional chunks from the same source document to be preserved as context.

Tech Stack

Python
Streamlit
Google Gemini
Sentence Transformers
FAISS
BM25
Cross-Encoder Reranking
PyPDF
NumPy
Project Structure
first-rag-based-agentic-ai/
│
├── main.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .env
├── docs/
│   ├── sample_rental_agreement.txt
│   ├── sample_nda.txt
│   └── ...
└── venv/

.env and venv/ should not be pushed to GitHub.

Setup

1. Clone the repository
git clone <your-repository-url>
cd first-rag-based-agentic-ai
2. Create a virtual environment
python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file:

GOOGLE_API_KEY=your_google_api_key
DOCS_PATH=docs
5. Add documents

Place .txt or .pdf files inside:

docs/
Running the Application
streamlit run main.py

Then open the local URL displayed by Streamlit.

Evaluation

The project includes evaluate.py, which tests whether the retrieval pipeline returns the expected source document for a set of manually curated queries.

Run:

python evaluate.py

Example output:

Test 1
Query: What is the monthly rent?
Expected source: sample_rental_agreement.txt
Retrieved sources: ['sample_rental_agreement.txt']
Result: PASS

Example evaluation result:

Total tests: 6
Passed: 6
Failed: 0
Retrieval Accuracy: 100.00%

Note: This result is based on a small manually curated evaluation set of six queries. It measures source-document retrieval accuracy and should not be interpreted as overall RAG system accuracy.

Future Improvements
Larger evaluation dataset.
Answer-level evaluation.
Support for additional document formats.
Persistent vector storage.
Conversation memory.
Hybrid retrieval tuning.
Deployment using Docker or a cloud platform.
Key Learning Outcomes

This project demonstrates practical implementation of:

Retrieval-Augmented Generation
Hybrid search
Dense vector retrieval
Keyword-based retrieval
FAISS indexing
Ensemble retrieval
Cross-Encoder reranking
Document chunking
Context selection
LLM-based answer generation
Retrieval evaluation
