import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)


def tokenize_text(text: str) -> List[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[str]:

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_length = len(sentence)

        if (
            current_chunk
            and current_length + sentence_length + 1 > chunk_size
        ):
            chunks.append(
                " ".join(current_chunk)
            )

            overlap_text = " ".join(current_chunk)
            overlap_text = overlap_text[-overlap:]

            current_chunk = [overlap_text]
            current_length = len(overlap_text)

        current_chunk.append(sentence)
        current_length += sentence_length + 1

    if current_chunk:
        chunks.append(
            " ".join(current_chunk)
        )

    return chunks

def load_documents(docs_path: str) -> List[Document]:
    documents: List[Document] = []
    root = Path(docs_path)

    for path in sorted(root.rglob("*")):

        if path.suffix.lower() == ".txt":
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).strip()
            except Exception:
                continue

            chunks = chunk_text(text)

            for chunk_index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(path),
                            "chunk": str(chunk_index),
                        },
                    )
                )

        elif path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(path)

                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""

                    chunks = chunk_text(text)

                    for chunk_index, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": str(path),
                                    "page": str(page_number),
                                    "chunk": str(chunk_index),
                                },
                            )
                        )

            except Exception:
                continue

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
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def get_relevant_documents(
        self,
        query: str,
        min_score: float = 0.25
    ) -> List[Document]:
        if not query.strip():
            return []

        query_embedding = self._embed_texts([query])
        distances, indices = self.index.search(query_embedding, self.k)

        results: List[Document] = []

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            if float(score) < min_score:
                continue

            doc = self.documents[idx]

            metadata = {
                **doc.metadata,
                "score": f"{float(score):.4f}"
            }

            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            )

        return results


class EnsembleRetriever:
    def __init__(
        self,
        retrievers: List[object],
        weights: Optional[List[float]] = None,
        k: int = 3,
        rrf_k: int = 60,
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.k = k
        self.rrf_k = rrf_k

    @staticmethod
    def _doc_id(doc: Document, fallback_index: int) -> str:
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get(
            "chunk",
            doc.metadata.get("chunk_id", fallback_index)
        )
        return f"{source}::{chunk}"

    def get_relevant_documents(self, query: str) -> List[Document]:
        fused_scores: Dict[str, float] = {}
        documents: Dict[str, Document] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.get_relevant_documents(query)

            for rank, doc in enumerate(results, start=1):
                doc_id = self._doc_id(doc, rank)

                rrf_score = weight / (self.rrf_k + rank)

                fused_scores[doc_id] = (
                    fused_scores.get(doc_id, 0.0) + rrf_score
                )

                if doc_id not in documents:
                    documents[doc_id] = doc

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )

        results = []

        for doc_id in ranked_ids:
            doc = documents[doc_id]

            metadata = {
                **doc.metadata,
                "ensemble_score": f"{fused_scores[doc_id]:.6f}",
            }

            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            )

            if len(results) >= self.k:
                break

        return results
class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        k: int = 3,
        max_score_gap: float = 4.0,
        max_chunks_per_source: int = 2,
    ):
        self.model = CrossEncoder(model_name)
        self.k = k
        self.max_score_gap = max_score_gap
        self.max_chunks_per_source = max_chunks_per_source

    def get_relevant_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:

        if not documents:
            return []

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        best_score = float(ranked[0][1])
        best_source = ranked[0][0].metadata.get(
            "source",
            "unknown"
        )

        results = []
        source_counts = {}

        for doc, score in ranked:
            score = float(score)
            source = doc.metadata.get("source", "unknown")

            # Always keep the best result.
            if not results:
                keep = True

            # Allow another chunk from the same document
            # as the best result.
            elif source == best_source:
                keep = True

            # For other documents, only keep them if
            # their score is close enough to the best score.
            elif (best_score - score) <= self.max_score_gap:
                keep = True

            else:
                keep = False

            if not keep:
                continue

            # Do not take too many chunks from one document.
            if source_counts.get(source, 0) >= self.max_chunks_per_source:
                continue

            metadata = {
                **doc.metadata,
                "reranker_score": f"{score:.4f}",
            }

            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

            source_counts[source] = (
                source_counts.get(source, 0) + 1
            )

            if len(results) >= self.k:
                break

        return results

def configure_gemini(api_key: str):
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY must be set in the environment."
        )

    return genai.Client(api_key=api_key)


def generate_answer(
    client,
    prompt: str,
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.2,
) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        ),
    )

    return response.text


def build_rag_resources(docs_path: str, google_api_key: str):
    documents = load_documents(docs_path)

    if not documents:
        raise ValueError(
            f"No documents found in {docs_path}. "
            "Place .txt or .pdf files inside the docs folder."
        )

    client = configure_gemini(google_api_key)

    bm25_retriever = BM25Retriever(documents, k=5)
    faiss_retriever = FaissRetriever(documents, k=5)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7],
        k=5
    )

    reranker = Reranker(k=3)

    return ensemble_retriever, reranker, client


def build_prompt(query: str, documents: List[Document]) -> str:
    prompt_lines = [
        "You are a helpful assistant. Use the information from the retrieved document excerpts to answer the user's question.",
        "Cite the source path when relevant and answer concisely.",
        "",
    ]

    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        chunk = doc.metadata.get("chunk")

        source_info = f"Source: {Path(source).name}"

        if page:
            source_info += f"\nPage: {page}"

        if chunk is not None:
            source_info += f"\nChunk: {chunk}"

        excerpt = doc.page_content.strip()

        if len(excerpt) > 1200:
            excerpt = excerpt[:1200].rstrip() + "..."

        prompt_lines.append(
            f"{source_info}\n\n{excerpt}\n"
        )

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
                retriever, reranker, client = get_resources(
                    docs_path,
                    google_api_key
                )
            except Exception as exc:
                st.error(f"Error creating RAG resources: {exc}")
                return

        with st.spinner("Retrieving relevant source chunks..."):
            retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            st.warning("No relevant documents found for your question.")
            return

        with st.spinner("Reranking retrieved chunks..."):
            retrieved_docs = reranker.get_relevant_documents(
                query,
                retrieved_docs
            )

        if not retrieved_docs:
            st.warning("No relevant documents remained after reranking.")
            return

        prompt = build_prompt(query, retrieved_docs)

        try:
            with st.spinner("Generating answer with Gemini 3.5 Flash..."):
                answer = generate_answer(
                    client,
                    prompt
                )
        except Exception as exc:
            st.error(f"Error generating answer: {exc}")
            return

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source chunks")

        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            chunk = doc.metadata.get("chunk")
            reranker_score = doc.metadata.get("reranker_score", "N/A")

            source_name = Path(source).name

            label = f"Source {idx}: {source_name}"

            if page:
                label += f" | Page {page}"

            if chunk is not None:
                label += f" | Chunk {chunk}"

            label += f" | Reranker Score: {reranker_score}"

            content = doc.page_content.strip()

            with st.expander(label):
                st.write(content)

    else:
        st.info("Enter a query above to start the RAG retrieval process.")


if __name__ == "__main__":
    main()
