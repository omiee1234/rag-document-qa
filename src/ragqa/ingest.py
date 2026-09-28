"""Ingest pipeline: load documents -> chunk -> embed -> store -> persist to a single index file."""

import pickle
from pathlib import Path

from .chunking import chunk_structured
from .embeddings import Embedder, get_embedder
from .loaders import load_structured_documents
from .store import InMemoryVectorStore
from .structure import StructuredDoc, parse_markdown

# Bump when the chunking/embedding scheme changes, so an index built by an
# older version is rebuilt rather than silently mixed with new queries.
INDEX_FORMAT_VERSION = 2


def build_index_from_documents(
    documents: dict[str, str | StructuredDoc] | list[StructuredDoc],
    embedder_name: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 150,
) -> tuple[Embedder, InMemoryVectorStore]:
    """Chunk + embed + store documents, entirely in memory.

    Accepts parsed StructuredDocs, or {doc_id: text} where text is parsed as
    markdown/plain text. No disk I/O -- this is what lets the UI build a
    per-user index from uploaded files without writing them anywhere, and
    without touching the shared on-disk index.pkl other sessions read.
    """
    if isinstance(documents, dict):
        docs = [
            d if isinstance(d, StructuredDoc) else parse_markdown(doc_id, d)
            for doc_id, d in documents.items()
        ]
    else:
        docs = list(documents)
    if not docs:
        raise ValueError("No documents to index")

    chunks = [c for doc in docs for c in chunk_structured(doc, chunk_size=chunk_size, overlap=overlap)]
    if not chunks:
        raise ValueError("No text could be extracted (scanned/image-only PDFs aren't supported)")

    embedder = get_embedder(embedder_name)
    texts = [c.embed_text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = InMemoryVectorStore()
    for chunk, vector in zip(chunks, vectors):
        store.add(chunk.chunk_id, chunk.doc_id, chunk.text, vector, chunk.title, chunk.section, chunk.page)

    return embedder, store


def ingest(
    doc_dir: str,
    index_path: str,
    embedder_name: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 150,
) -> InMemoryVectorStore:
    documents = load_structured_documents(Path(doc_dir))
    if not documents:
        raise ValueError(f"No supported documents (.pdf/.txt/.md) found in {doc_dir}")

    embedder, store = build_index_from_documents(
        documents, embedder_name=embedder_name, chunk_size=chunk_size, overlap=overlap
    )

    # embedder is persisted alongside the store: a TF-IDF vectorizer's vocabulary
    # is fit to this corpus, so queries later must reuse the exact same transform.
    with open(index_path, "wb") as f:
        pickle.dump({"version": INDEX_FORMAT_VERSION, "embedder": embedder, "store": store}, f)

    print(f"Ingested {len(store)} chunks from {len(documents)} document(s) -> {index_path}")
    return store
