"""
PubMedQA Hybrid RAG — local long-term app
Run: uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_BM25 = 8
TOP_K_DENSE = 8
TOP_K_FINAL = 5
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | gemini | none
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_DOCS = int(os.getenv("MAX_DOCS", "2500"))  # keep first-run light

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "web")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    verdict: Optional[str]
    hits: List[Dict[str, Any]]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def parse_verdict(text: str) -> Optional[str]:
    m = re.search(r"Verdict:\s*(Yes|No|Maybe)", text, flags=re.I)
    return m.group(1).lower() if m else None


class RagEngine:
    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.ready = False

    def build(self) -> None:
        raw = load_dataset("pubmed_qa", "pqa_labeled")
        rows = list(raw["train"])

        docs: List[Dict[str, Any]] = []
        for qa_idx, row in enumerate(rows):
            for ctx_i, passage in enumerate(row["context"]["contexts"]):
                docs.append(
                    {
                        "doc_id": f"{qa_idx}-{ctx_i}",
                        "text": (passage or "").strip(),
                        "question": row["question"],
                    }
                )
                if len(docs) >= MAX_DOCS:
                    break
            if len(docs) >= MAX_DOCS:
                break

        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        embs = self.embedder.encode(
            [d["text"] for d in docs],
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embs = np.asarray(embs, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.ready = True

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(tokenize(query))
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0]

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        assert self.embedder is not None and self.index is not None
        qv = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True), dtype=np.float32
        )
        scores, idxs = self.index.search(qv, top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if int(i) >= 0]

    @staticmethod
    def rrf(
        result_lists: List[List[Tuple[int, float]]], k: int = 60, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        fused: Dict[int, float] = {}
        for results in result_lists:
            for rank, (doc_i, _) in enumerate(results, start=1):
                fused[doc_i] = fused.get(doc_i, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def retrieve(self, query: str, mode: str, top_k: int) -> List[Dict[str, Any]]:
        bm = self.bm25_search(query, TOP_K_BM25)
        de = self.dense_search(query, TOP_K_DENSE)
        if mode == "bm25":
            ranked = bm[:top_k]
        elif mode == "dense":
            ranked = de[:top_k]
        else:
            ranked = self.rrf([bm, de], top_k=top_k)

        out = []
        for doc_i, score in ranked:
            d = self.docs[doc_i]
            out.append({"doc_id": d["doc_id"], "score": score, "text": d["text"]})
        return out


SYSTEM_PROMPT = """You are a careful biomedical research assistant.
Answer ONLY using the provided evidence passages.
Write a detailed answer (5-10 sentences) explaining the findings.
Cite evidence as [1], [2], etc.
End with exactly one line in this format:
Verdict: Yes
or
Verdict: No
or
Verdict: Maybe
If evidence is insufficient or conflicting, choose Maybe.
If the question is not yes/no, still give a detailed grounded answer and use Maybe unless evidence clearly supports a yes/no claim."""


def generate_answer(question: str, passages: List[str]) -> str:
    blocks = [f"[{i}] {p}" for i, p in enumerate(passages, 1)]
    prompt = (
        f"Question: {question}\n\nEvidence passages:\n"
        + "\n\n".join(blocks)
        + "\n\nWrite a detailed answer with citations, then the Verdict line."
    )

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()

    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()

    joined = " ".join(passages)[:1000]
    return (
        f"Based on retrieved evidence for '{question}': {joined}...\n"
        f"Configure GROQ_API_KEY or GEMINI_API_KEY for full LLM answers.\n"
        f"Verdict: Maybe"
    )


@lru_cache(maxsize=1)
def get_engine() -> RagEngine:
    eng = RagEngine()
    eng.build()
    return eng


app = FastAPI(title="PubMedQA Hybrid RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Build index once at boot (first run downloads dataset + model)
    get_engine()


@app.get("/api/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    return {
        "ok": eng.ready,
        "docs": len(eng.docs),
        "provider": LLM_PROVIDER,
        "embed_model": EMBED_MODEL,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    eng = get_engine()
    if not eng.ready:
        raise HTTPException(503, "Engine not ready")
    hits = eng.retrieve(req.question.strip(), req.mode, req.top_k)
    if not hits:
        raise HTTPException(404, "No evidence found")
    answer = generate_answer(req.question.strip(), [h["text"] for h in hits])
    return AskResponse(
        question=req.question.strip(),
        mode=req.mode,
        answer=answer,
        verdict=parse_verdict(answer),
        hits=hits,
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
