"""Load raw text out of source documents (PDF/TXT/MD)."""

from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_document(path: Path) -> str:
    """Return the raw text contents of a single document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def load_documents(doc_dir: Path) -> dict[str, str]:
    """Load every supported document in a directory. Returns {doc_id: text}."""
    doc_dir = Path(doc_dir)
    texts: dict[str, str] = {}
    for path in sorted(doc_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            texts[path.stem] = load_document(path)
    return texts


def load_document_bytes(filename: str, data: bytes) -> str:
    """Same as load_document, but from in-memory bytes (e.g. a browser upload)
    instead of a filesystem path -- avoids ever writing uploaded content to disk."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix} ({filename})")
