"""Ingest pipeline: load documents -> chunk -> embed -> store -> persist to a single index file."""

import pickle
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import Embedder, get_embedder
from .loaders import load_documents
from .store import InMemoryVectorStore


def build_index_from_documents(
    documents: dict[str, str],
    embedder_name: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 150,
) -> tuple[Embedder, InMemoryVectorStore]:
    """Chunk + embed + store a set of {doc_id: text} documents, entirely in memory.

    No disk I/O -- this is what lets the UI build a per-user index from
    uploaded files without writing them anywhere, and without touching the
    shared on-disk index.pkl that other users' sessions also read.
    """
    if not documents:
        raise ValueError("No documents to index")

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    embedder = get_embedder(embedder_name)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = InMemoryVectorStore()
    for chunk, vector in zip(chunks, vectors):
        store.add(chunk.chunk_id, chunk.doc_id, chunk.text, vector)

    return embedder, store


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

    embedder, store = build_index_from_documents(
        documents, embedder_name=embedder_name, chunk_size=chunk_size, overlap=overlap
    )

    # embedder is persisted alongside the store: a TF-IDF vectorizer's vocabulary
    # is fit to this corpus, so queries later must reuse the exact same transform.
    with open(index_path, "wb") as f:
        pickle.dump({"embedder": embedder, "store": store}, f)

    print(f"Ingested {len(store)} chunks from {len(documents)} document(s) -> {index_path}")
    return store
