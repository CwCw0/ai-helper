import time, os, math, json, hashlib
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm
import re
import openai

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

class LocalEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

import uuid

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for v, m in zip(vectors, metadatas):
            h = m.get("hash", "")
            try:
                point_id = uuid.UUID(h[:32])
            except ValueError:
                point_id = uuid.UUID(bytes=bytes.fromhex(h[:16].ljust(16, '0')))
            vec_list = [float(x) for x in v]
            payload = {
                k: (v if isinstance(v, (str, int, float, list, dict, bool, type(None))) else str(v))
                for k, v in m.items()
            }
            points.append(qm.PointStruct(id=str(point_id), vector=vec_list, payload=payload))


        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OpenAILLM:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section when relevant.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
  
    def __init__(self):
        # ---- Embedding ----
        if settings.openai_api_key:
            try:
                self.embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
            except Exception as e:
                print("[WARN] Failed to init OpenAIEmbeddings:", e)
                self.embedder = LocalEmbedder(dim=384)
        else:
            self.embedder = LocalEmbedder(dim=384)

        # ---- Vector store ----
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # ---- LLM selection ----
        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                self.llm = OpenAILLM(api_key=settings.openai_api_key)
                self.llm_name = "openai:gpt-4o-mini"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        # ---- Metrics & counters ----
        self.metrics = Metrics()

        # ---- Document tracking ----
        self._doc_titles = set()
        self._chunk_count = 0


    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
    # Embed query and search top-k from vector store
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k=k)
        return [meta for score, meta in results]


    def keyword_filter(self, query: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
        q_tokens = set(query.lower().split())
        scored_chunks = []
        for chunk in chunks:
            c_tokens = set(chunk.get("text", "").lower().split())
            score = len(q_tokens & c_tokens)
            if score > 0:
                scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for score, c in scored_chunks[:top_k]]
        return top_chunks


    def generate(self, query: str, contexts: List[Dict] = None) -> str:
            t0 = time.time()

            # Step 1: Retrieve top chunks if none provided
            if contexts is None:
                contexts = self.retrieve(query, k=6)

            # Step 2: Optional keyword boost (hybrid approach)
            q_tokens = set(query.lower().split())
            scored_chunks = []
            for c in contexts:
                c_tokens = set(c.get("text", "").lower().split())
                score = len(q_tokens & c_tokens)
                scored_chunks.append((score, c))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            # Keep top 4 most relevant chunks by keyword score, fallback to embedding top if all zero
            top_chunks = [c for score, c in scored_chunks if score > 0]
            if not top_chunks:
                top_chunks = contexts[:4]

            # Step 3: Generate answer with LLM
            answer = self.llm.generate(query, top_chunks)
            self.metrics.add_generation((time.time() - t0) * 1000.0)
            return answer



    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m
        }

# ---- Helpers ----
# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int = 700, chunk_overlap: int = 80) -> List[Dict]:
 
    chunks = []

    for doc in docs:
        text = doc.get("text", "")
        title = doc.get("title", "")
        print(f"\n[DEBUG] Processing document: {title}")  # <-- show doc title
        
        # Split text by headings, keeping the headings
        sections = re.split(r'(#+ .+)', text)
        current_section = ""
        current_text = ""

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            if sec.startswith("#"):  # Heading line
                # Append the previous section's text
                if current_text:
                    start = 0
                    while start < len(current_text):
                        end = min(start + chunk_size, len(current_text))
                        chunk_text = current_text[start:end].strip()
                        if chunk_text:
                            chunks.append({
                                "title": title,
                                "section": current_section,
                                "text": chunk_text
                            })
                            print(f"[DEBUG] Added chunk from section: {current_section[:50]}...")  # <-- debug section
                            print(f"[DEBUG] Chunk text preview: {chunk_text[:80]}...\n")  # preview first 80 chars
                        start += chunk_size - chunk_overlap
                    current_text = ""

                current_section = sec  # Update to new heading
                print(f"[DEBUG] New section found: {current_section}")  # <-- debug new heading
            else:
                current_text += sec + "\n"

        # Append the last section
        if current_text:
            start = 0
            while start < len(current_text):
                end = min(start + chunk_size, len(current_text))
                chunk_text = current_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "title": title,
                        "section": current_section,
                        "text": chunk_text
                    })
                    print(f"[DEBUG] Added last chunk from section: {current_section[:50]}...")
                    print(f"[DEBUG] Chunk text preview: {chunk_text[:80]}...\n")
                start += chunk_size - chunk_overlap

    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    return chunks
