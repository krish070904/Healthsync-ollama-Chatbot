# app/rag.py  (replace your current file with this exact content)
import os
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# CrossEncoder is helpful but may fail on some systems; guard it
try:
    from sentence_transformers import CrossEncoder
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"[RAG] CrossEncoder not available: {e}")
    cross = None

from .config import (
    DATA_DIR,
    DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_CONTEXT_CHARS,
    OLLAMA_MODEL,
)


class RetrievedDoc(BaseModel):
    source: str
    page: int
    score: float
    snippet: str


class RAGPipeline:
    def __init__(self):
        print("[RAG] Initializing embeddings...")
        self.embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db = None

    def ingest(self) -> None:
        """Read PDFs, TXTs, and CSVs recursively from DATA_DIR and build Chroma DB"""
        print(f"[RAG] Ingesting documents from: {DATA_DIR}")
        docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

        # Recursively find PDFs
        pdf_paths = list(sorted(Path(DATA_DIR).rglob("*.pdf")))
        print(f"[RAG] Found {len(pdf_paths)} PDF(s).")
        for pdf in pdf_paths:
            try:
                print(f"[RAG] Reading PDF: {pdf}")
                loader = PyPDFLoader(str(pdf))
                pages = loader.load()
                for p in pages:
                    p.metadata.setdefault("source", pdf.name)
                splits = splitter.split_documents(pages)
                docs.extend(splits)
            except Exception as e:
                print(f"[RAG] ERROR reading PDF {pdf}: {e}")

        # TXT files
        txt_paths = list(sorted(Path(DATA_DIR).rglob("*.txt")))
        print(f"[RAG] Found {len(txt_paths)} TXT file(s).")
        for txt in txt_paths:
            try:
                print(f"[RAG] Reading TXT: {txt}")
                loader = TextLoader(str(txt), encoding="utf-8")
                pages = loader.load()
                for p in pages:
                    p.metadata.setdefault("source", txt.name)
                splits = splitter.split_documents(pages)
                docs.extend(splits)
            except Exception as e:
                print(f"[RAG] ERROR reading TXT {txt}: {e}")

        # CSV files
        csv_paths = list(sorted(Path(DATA_DIR).rglob("*.csv")))
        print(f"[RAG] Found {len(csv_paths)} CSV file(s).")
        for csv in csv_paths:
            try:
                print(f"[RAG] Reading CSV: {csv}")
                loader = CSVLoader(str(csv))
                pages = loader.load()
                for p in pages:
                    p.metadata.setdefault("source", csv.name)
                splits = splitter.split_documents(pages)
                docs.extend(splits)
            except Exception as e:
                print(f"[RAG] ERROR reading CSV {csv}: {e}")

        if not docs:
            raise RuntimeError(f"[RAG] No documents were ingested from {DATA_DIR}. Please add PDFs/CSVs/TXTs.")

        print(f"[RAG] Total document chunks to index: {len(docs)}")
        # Create Chroma vector DB
        self.db = Chroma.from_documents(documents=docs, embedding=self.embed, persist_directory=str(DB_DIR))
        print("[RAG] Chroma DB built, skipping persist() for speed.")

        

    def load_or_create(self) -> None:
        """Load existing DB or create it if missing or empty."""
        # If DB_DIR doesn't have files -> force ingest
        has_files = any(DB_DIR.iterdir()) if DB_DIR.exists() else False
        if not has_files:
            print("[RAG] Storage directory empty — running ingest()")
            self.ingest()
            return

        try:
            print("[RAG] Loading existing Chroma DB...")
            self.db = Chroma(persist_directory=str(DB_DIR), embedding_function=self.embed)
            # Quick sanity check
            results = self.db.similarity_search("health", k=1)
            if not results:
                print("[RAG] Loaded DB but no results in sanity check — re-ingesting.")
                self.ingest()
        except Exception as e:
            print(f"[RAG] Error loading Chroma DB: {e}. Running ingest() to rebuild.")
            self.ingest()

    def retrieve(self, question: str, k: int = TOP_K) -> List[RetrievedDoc]:
        if not self.db:
            raise RuntimeError("[RAG] DB not initialized. Call load_or_create() first.")

        print(f"[RAG] Retrieving for question: {question}")
        try:
            candidates = self.db.similarity_search_with_relevance_scores(question, k=TOP_K * 3)
        except Exception as e:
            print(f"[RAG] similarity_search_with_relevance_scores error: {e}")
            candidates = []

        if not candidates:
            print("[RAG] No candidate documents returned by similarity search.")
            return []

        # Try rerank with cross-encoder if available
        try:
            if cross is not None:
                print("[RAG] Reranking with CrossEncoder...")
                pairs = [(question, doc.page_content) for doc, _ in candidates]
                scores = cross.predict(pairs)
                reranked = sorted(
                    [(doc, float(score)) for (doc, _), score in zip(candidates, scores)],
                    key=lambda x: x[1],
                    reverse=True,
                )[:k]
            else:
                print("[RAG] CrossEncoder not available; using raw similarity scores.")
                reranked = [(doc, float(score)) for doc, score in candidates][:k]
        except Exception as e:
            print(f"[RAG] Cross-encoder failed: {e}. Falling back to raw similarity scores.")
            reranked = [(doc, float(score)) for doc, score in candidates][:k]

        items: List[RetrievedDoc] = []
        for doc, score in reranked:
            src = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0)) + 1
            snippet = doc.page_content[:800].strip()
            items.append(RetrievedDoc(source=src, page=page, score=score, snippet=snippet))
        print(f"[RAG] Returning {len(items)} retrieved items.")
        return items

    def _build_prompt(self, question: str, retrieved: List[RetrievedDoc]) -> str:
        context_blocks = []
        total_chars = 0
        for i, d in enumerate(retrieved, start=1):
            chunk = f"[{i}] (source: {d.source}, page {d.page})\n{d.snippet}\n"
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break
            context_blocks.append(chunk)
            total_chars += len(chunk)

        context_str = "\n\n".join(context_blocks)
        system = (
            "You are a strict medical information assistant. "
            "Use ONLY the provided sources. Do NOT hallucinate. "
            'If the sources do not clearly answer, reply: "I don\'t know; consult a clinician." '
            "Always cite sources with [1], [2]… and include a final bulleted 'Sources' list."
        )
        user = f"""Question: {question}

Sources:
{context_str}

Instructions:
- Provide a concise, factual answer.
- Include a short rationale with inline citations like [1].
- End with a Sources list that maps citations to filenames (and page if available)."""
        return f"{system}\n\n{user}"

    def generate(self, prompt: str) -> str:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 8192, "temperature": 0.0, "max_tokens": 512},
        )
        return resp.get("message", {}).get("content", "").strip()

    @staticmethod
    def confidence_from_scores(scores: List[float]) -> float:
        if not scores:
            return 0.0
        mx = max(scores)
        avg = sum(scores) / len(scores)
        conf = 0.6 * avg + 0.4 * mx
        return max(0.0, min(1.0, conf))

    def ask(self, question: str) -> Dict[str, Any]:
        retrieved = self.retrieve(question)
        # If nothing retrieved — avoid calling the model and return low-confidence message
        if not retrieved:
            return {"answer": "I don't have relevant documents indexed to answer that question. Please add trusted guidelines or datasets and try again.", "confidence": 0.0, "sources": []}

        prompt = self._build_prompt(question, retrieved)
        output = self.generate(prompt)
        scores = [d.score for d in retrieved]
        conf = self.confidence_from_scores(scores)
        sources_list = [{"source": d.source, "page": d.page, "score": d.score} for d in retrieved]
        return {"answer": output, "confidence": round(conf, 3), "sources": sources_list}
