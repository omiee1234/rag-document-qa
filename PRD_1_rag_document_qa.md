# PRD 1: RAG Document Q&A System with Measured Retrieval Quality

## Problem
Anyone can build a RAG demo that "looks right." Almost no junior candidate can show *measured* retrieval quality. This project proves you can evaluate a RAG pipeline, not just wire one together with LangChain.

## Goal
Upload documents, ask questions, get answers with citations to the source chunk — and a dashboard showing recall@k, MRR and answer faithfulness on a held-out question set.

## Users / use case
A small business or legal/HR team uploads its policy documents and asks natural-language questions instead of searching manually. (This is also directly relevant to the Upwork "Legal Document Intelligence" job.)

## Scope (v1 — 1.5 to 2 weeks)
**In scope**
- Ingest PDFs/TXT/MD, chunk with overlap (configurable size)
- Provider-agnostic embeddings: TF-IDF (offline, no API key) as default, swappable for OpenAI/Gemini/sentence-transformers embeddings via one config value
- Vector store: start in-memory (numpy cosine similarity), documented swap-in for pgvector/Qdrant
- Retrieval: top-k with a similarity threshold
- Answer generation: cite the exact chunk(s) used; provider-agnostic LLM call (reuse the pattern from `llm-eval-harness`)
- **Evaluation harness** (the differentiator): a labeled question set (question → correct chunk ids), computing:
  - Recall@k, Precision@k, MRR (retrieval quality)
  - Faithfulness check: does the generated answer's claims actually appear in the retrieved chunks? (simple entailment/overlap check, or LLM-as-judge)
- CLI + a minimal FastAPI endpoint (`/ask`)
- README with an architecture diagram and a results table

**Out of scope (v1)**
- Multi-user auth, a UI beyond a simple form, multi-language documents, OCR for scanned PDFs

## Tech stack
Python, `pypdf`/`pdfplumber`, scikit-learn (TF-IDF baseline), numpy, FastAPI, pytest, optional: sentence-transformers, pgvector/Qdrant, Docker

## Data
10–20 sample documents you choose (e.g. a company HR policy set, or public docs like a product manual). Hand-write 20–30 question/answer pairs with the correct source chunk marked — this labeled set is what makes the evaluation real, not synthetic.

## Milestones
1. Chunking + TF-IDF embedding + in-memory store working end to end (CLI: ingest → query → get chunk back)
2. Answer generation with citations
3. Evaluation harness with recall@k/MRR/faithfulness, run against the labeled set, get a baseline number
4. Swap in one real embedding provider, compare metrics against the TF-IDF baseline
5. FastAPI wrapper + Docker + README with results table and an architecture diagram

## Evaluation (how you'll know it's good)
A results table in the README, e.g.:
| Embedding | Recall@3 | MRR | Faithfulness |
|---|---|---|---|
| TF-IDF baseline | 0.62 | 0.55 | 91% |
| sentence-transformers | 0.81 | 0.74 | 95% |

## What this proves in an interview
- You understand retrieval is a search problem with measurable quality, not "just embed and hope"
- You can design a labeled eval set and defend a metric choice
- You can compare a cheap baseline against a stronger method honestly (same habit as your `llm-eval-harness` work)

## Resume line
"Built a RAG pipeline with a provider-agnostic embedding layer; measured recall@k, MRR and answer faithfulness against a labeled question set, improving recall@3 from 0.62 (TF-IDF) to 0.81 (embeddings)."
