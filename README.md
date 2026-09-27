# RAG Document Q&A with a Measured Evaluation Harness

Upload policy documents, ask natural-language questions, get answers with
citations to the exact source chunk — and, unlike most RAG demos, a
harness that actually *measures* whether the retrieval and the answers are
any good, against a hand-labeled question set.

## Architecture

```
                    ┌─────────────┐
  docs/*.md,.pdf →  │  loaders.py │ → raw text per document
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ chunking.py │ → overlapping Chunk objects, stable ids ("doc::0")
                    └─────────────┘
                          │
                          ▼
                    ┌───────────────┐        swappable via --embedder:
                    │ embeddings.py │  ←──   tfidf | sentence-transformers | openai
                    └───────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  store.py   │  in-memory numpy cosine similarity
                    └─────────────┘  (swap-in path: pgvector / Qdrant)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     ┌──────────────┐           ┌──────────────┐
     │ retrieve.py  │           │ evaluate.py  │ ← data/eval/qa_set.json
     │ top-k search │           │ recall/MRR/  │   (hand-labeled ground truth)
     └──────────────┘           │ faithfulness │
            │                   └──────────────┘
            ▼
     ┌──────────────┐
     │ generate.py  │ → answer + citations (LLM if OPENAI_API_KEY set, else extractive)
     └──────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
  cli.py       api.py (FastAPI: POST /ask)
```

Every arrow that says "swappable" really is: changing embedding provider or
vector store is a one-line/one-module change, everything downstream
(retrieval, generation, evaluation) is unaffected.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

Optional extras:
```bash
pip install -e ".[sentence-transformers]"   # real dense embeddings
pip install -e ".[openai]"                   # LLM-generated answers instead of extractive
```

## Usage

```bash
# 1. Ingest the sample HR policy docs into a TF-IDF index (no API key needed)
ragqa ingest --docs data/docs --index index.pkl --embedder tfidf

# 2. Ask a question
ragqa query "How many days of PTO do I get per year?" --index index.pkl

# 3. Score retrieval + faithfulness against the labeled question set
ragqa evaluate --qa-set data/eval/qa_set.json --index index.pkl --quiet

# 4. Serve it
export RAGQA_INDEX_PATH=index.pkl        # Windows: set RAGQA_INDEX_PATH=index.pkl
uvicorn ragqa.api:app --reload
# POST http://127.0.0.1:8000/ask  {"question": "..."}
```

Run the test suite:
```bash
pytest tests/ -q
```

## Web UI

A small Streamlit app on top of the same pipeline — ask a question, see the
grounded answer with its cited source chunks, or run the evaluation harness
with one click and see recall@k/precision@k/MRR/faithfulness as live metrics.
Free, local, no API key required.

```bash
pip install -e ".[ui]"
streamlit run src/ragqa/ui.py
# opens at http://localhost:8501
```

Two tabs:
- **Ask a question** — a question box, the grounded answer, and expandable
  cited chunks with their similarity scores
- **Evaluation results** — one button, then Recall@3/Precision@3/MRR/
  Faithfulness as metric tiles plus a per-question results table

## Docker

```bash
docker build -t ragqa:latest .
docker run -d -p 8000:8000 --name ragqa ragqa:latest
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How many days of PTO do I get?"}'
```

The image ingests the sample docs at build time, so the container is ready
to serve as soon as it starts.

## Evaluation results

Measured against the 24-question labeled set in `data/eval/qa_set.json`
(top-k = 3):

| Embedding | Recall@3 | Precision@3 | MRR | Faithfulness |
|---|---|---|---|---|
| TF-IDF (baseline) | 1.00 | 0.33 | 1.00 | 1.00 |
| sentence-transformers (`all-MiniLM-L6-v2`) | 1.00 | 0.33 | 0.97 | 1.00 |

**Honest caveat:** the bundled sample corpus is 8 documents on clearly
distinct topics (leave, remote work, expenses, conduct, onboarding,
security, parental leave, termination), so both embedders saturate at
near-perfect retrieval — there's no ambiguity for either to fail on.
Precision@3 is capped at 0.33 by construction: each document is a single
chunk, so 1 correct chunk out of top-3 retrieved is the ceiling.

This is the expected result for a small, well-separated corpus, and it's
also exactly the harness's value: it *tells you* when a comparison isn't
discriminating, instead of hiding it behind a demo that "looks right." To
see a real gap between TF-IDF and dense embeddings, grow the corpus to
50+ documents with overlapping vocabulary (e.g. multiple similarly-worded
policies, or real messy source material) — TF-IDF degrades on lexical
ambiguity and synonymy in a way dense embeddings don't.

## Project layout

```
src/ragqa/       pipeline modules (see architecture diagram above)
data/docs/       sample source documents
data/eval/       labeled question -> correct_chunk_ids set
tests/           pytest unit tests for chunking, store, evaluation math
Dockerfile       builds the index at image build time, serves /ask
```
