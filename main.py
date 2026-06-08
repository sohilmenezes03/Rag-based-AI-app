import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import generativeai as genai
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)


def tokenize_text(text: str) -> List[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token]


def load_documents(docs_path: str) -> List[Document]:
    documents: List[Document] = []
    root = Path(docs_path)

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".txt":
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if text:
                documents.append(Document(page_content=text, metadata={"source": str(path)}))

        elif path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(path)
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages).strip()
            except Exception:
                continue
            if text:
                documents.append(Document(page_content=text, metadata={"source": str(path)}))

    return documents


class BM25Retriever:
    def __init__(self, documents: List[Document], k: int = 4):
        self.documents = documents
        self.k = k
        self.tokenized_documents = [tokenize_text(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def get_relevant_documents(self, query: str) -> List[Document]:
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][: self.k]
        results: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            doc = self.documents[idx]
            metadata = {**doc.metadata, "score": f"{score:.4f}"}
            results.append(Document(page_content=doc.page_content, metadata=metadata))

        return results


class FaissRetriever:
    def __init__(self, documents: List[Document], model_name: str = "all-MiniLM-L6-v2", k: int = 4):
        self.documents = documents
        self.k = k
        self.embedder = SentenceTransformer(model_name)
        self.embeddings = self._embed_texts([doc.page_content for doc in documents])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not query.strip():
            return []

        query_embedding = self._embed_texts([query])
        distances, indices = self.index.search(query_embedding, self.k)

        results: List[Document] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            metadata = {**doc.metadata, "score": f"{float(score):.4f}"}
            results.append(Document(page_content=doc.page_content, metadata=metadata))

        return results


class EnsembleRetriever:
    def __init__(self, retrievers: List[object], weights: Optional[List[float]] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    @staticmethod
    def _get_score(doc: Document) -> float:
        raw_score = doc.metadata.get("score", "0")
        try:
            return float(raw_score)
        except (ValueError, TypeError):
            return 0.0

    def get_relevant_documents(self, query: str) -> List[Document]:
        scored_documents: Dict[str, Document] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            for doc in retriever.get_relevant_documents(query):
                source = doc.metadata.get("source", "unknown")
                score = self._get_score(doc) * weight
                existing = scored_documents.get(source)
                if existing is None or score > self._get_score(existing):
                    merged_metadata = {**doc.metadata, "ensemble_score": f"{score:.4f}"}
                    scored_documents[source] = Document(page_content=doc.page_content, metadata=merged_metadata)

        sorted_docs = sorted(
            scored_documents.values(),
            key=lambda doc: self._get_score(doc),
            reverse=True,
        )
        return sorted_docs[:5]


def configure_gemini(api_key: str) -> None:
    if not api_key:
        raise ValueError("GOOGLE_API_KEY must be set in the environment.")
    genai.configure(api_key=api_key)


def generate_answer(prompt: str, model_name: str = "gemini-3.5-flash", temperature: float = 0.2) -> str:
    model = genai.GenerativeModel(model_name=model_name)
    generation_config = genai.GenerationConfig(temperature=temperature)
    response = model.generate_content(prompt, generation_config=generation_config)
    return getattr(response, "text", str(response))


def build_rag_resources(docs_path: str, google_api_key: str):
    documents = load_documents(docs_path)
    if not documents:
        raise ValueError(f"No documents found in {docs_path}. Place .txt or .pdf files inside the docs folder.")

    configure_gemini(google_api_key)
    bm25_retriever = BM25Retriever(documents, k=4)
    faiss_retriever = FaissRetriever(documents, k=4)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.6, 0.4])
    return ensemble_retriever


def build_prompt(query: str, documents: List[Document]) -> str:
    prompt_lines = [
        "You are a helpful assistant. Use the information from the retrieved document excerpts to answer the user's question.",
        "Cite the source path when relevant and answer concisely.",
        "",
    ]

    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        excerpt = doc.page_content.strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200].rstrip() + "..."
        prompt_lines.append(f"Source {idx}: {source}\n{excerpt}\n")

    prompt_lines.append(f"Question: {query}\n")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


@st.cache_resource
def get_resources(docs_path: str, google_api_key: str):
    return build_rag_resources(docs_path, google_api_key)


def main():
    load_dotenv()
    st.set_page_config(page_title="Python RAG Agent", layout="wide")
    st.title("Python RAG Application")
    st.markdown(
        "Use this interface to ask questions over your local `docs/` files. Supported formats: `.txt`, `.pdf`."
    )

    docs_path = os.getenv("DOCS_PATH", "docs")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")

    if not google_api_key:
        st.error("Missing `GOOGLE_API_KEY` in environment. Add it to .env and restart.")
        return

    if not Path(docs_path).exists():
        st.error(f"Docs path not found: {docs_path}. Create the folder and add .txt or .pdf files.")
        return

    query = st.text_input("Ask a question", value="", placeholder="Enter a question about your documents...")
    st.markdown("---")

    if query:
        with st.spinner("Loading documents and building the retriever..."):
            try:
                retriever = get_resources(docs_path, google_api_key)
            except Exception as exc:
                st.error(f"Error creating RAG resources: {exc}")
                return

        with st.spinner("Retrieving relevant source chunks..."):
            retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            st.warning("No relevant documents found for your question.")
            return

        prompt = build_prompt(query, retrieved_docs)

        with st.spinner("Generating answer with Gemini 3.5 Flash..."):
            answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source chunks")
        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            with st.expander(f"Source {idx}: {source}"):
                st.write(content)
    else:
        st.info("Enter a query above to start the RAG retrieval process.")


if __name__ == "__main__":
    main()
