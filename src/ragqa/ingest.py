"""Ingest pipeline: load documents -> chunk -> embed -> store -> persist to a single index file."""

import pickle
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import get_embedder
from .loaders import load_documents
from .store import InMemoryVectorStore


def ingest(
    doc_dir: str,
    index_path: str,
    embedder_name: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 150,
) -> InMemoryVectorStore:
    documents = load_documents(Path(doc_dir))
    if not documents:
        raise ValueError(f"No supported documents (.pdf/.txt/.md) found in {doc_dir}")

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    embedder = get_embedder(embedder_name)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = InMemoryVectorStore()
    for chunk, vector in zip(chunks, vectors):
        store.add(chunk.chunk_id, chunk.doc_id, chunk.text, vector)

    # embedder is persisted alongside the store: a TF-IDF vectorizer's vocabulary
    # is fit to this corpus, so queries later must reuse the exact same transform.
    with open(index_path, "wb") as f:
        pickle.dump({"embedder": embedder, "store": store}, f)

    print(f"Ingested {len(chunks)} chunks from {len(documents)} document(s) -> {index_path}")
    return store
