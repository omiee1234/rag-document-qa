"""Minimal FastAPI wrapper around the pipeline: POST /ask."""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from .generate import answer_question
from .retrieve import retrieve

INDEX_PATH = os.environ.get("RAGQA_INDEX_PATH", "index.pkl")

app = FastAPI(title="RAG Document Q&A")


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    citations: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    retrieved = retrieve(req.question, INDEX_PATH, top_k=req.top_k)
    answer, citations = answer_question(req.question, retrieved)
    return AskResponse(answer=answer, citations=citations)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
