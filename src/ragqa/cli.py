"""Command-line entrypoint: ingest -> query -> evaluate."""

import argparse
import json

from .evaluate import run_evaluation
from .generate import answer_question, low_confidence_threshold
from .ingest import ingest
from .retrieve import load_index, search_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragqa")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk + embed a folder of documents into an index")
    p_ingest.add_argument("--docs", required=True, help="Directory of .pdf/.txt/.md files")
    p_ingest.add_argument("--index", default="index.pkl", help="Output index file")
    p_ingest.add_argument(
        "--embedder", default="tfidf", choices=["tfidf", "hybrid", "sentence-transformers", "openai"]
    )
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--overlap", type=int, default=150)

    p_query = sub.add_parser("query", help="Ask a single question against an index")
    p_query.add_argument("question")
    p_query.add_argument("--index", default="index.pkl")
    p_query.add_argument("--top-k", type=int, default=5)

    p_eval = sub.add_parser("evaluate", help="Score retrieval + faithfulness against a labeled QA set")
    p_eval.add_argument("--qa-set", required=True)
    p_eval.add_argument("--index", default="index.pkl")
    # kept at 3 (not bumped to match `query`'s new default of 5): this is the
    # documented, measured default -- the README's Recall@3/Precision@3/MRR
    # numbers are specifically "at top-k=3" and would silently go stale if
    # this changed without re-running and updating them.
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.add_argument("--quiet", action="store_true", help="Hide per-question detail")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(
            args.docs,
            args.index,
            embedder_name=args.embedder,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    elif args.command == "query":
        embedder, store = load_index(args.index)
        retrieved = search_index(embedder, store, args.question, top_k=args.top_k)
        answer, citations = answer_question(args.question, retrieved)
        top_score = retrieved[0][1] if retrieved else 0.0
        if top_score < low_confidence_threshold(embedder):
            print(
                f"Warning: low-confidence match (top score: {top_score:.3f}). "
                "This index may not actually contain a good answer to this question."
            )
        if retrieved:
            top = retrieved[0][0]
            where = " > ".join(p for p in dict.fromkeys((top.title, top.section)) if p)
            print("Source:", f"{where} (page {top.page})" if top.page else where or top.doc_id)
        print("Answer:", answer)
        print("Citations:", citations)

    elif args.command == "evaluate":
        results = run_evaluation(args.qa_set, args.index, top_k=args.top_k)
        if args.quiet:
            results = {k: v for k, v in results.items() if k != "per_question"}
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
