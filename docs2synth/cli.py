"""Command-line interface for Docs2Synth.

Currently, this is a placeholder that will later expose key functionality.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docs2synth",
        description="Command-line utilities for the Docs2Synth project.",
    )

    parser.add_argument(
        "--version", "-V", action="version", version="%(prog)s 0.1.0"
    )

    # Future sub-commands will be registered here.

    subparsers = parser.add_subparsers(dest="command", required=False)

    # --- generate-qa ------------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate-qa",
        help="Generate question-answer pairs from a document corpus.",
    )
    gen_parser.add_argument("input", help="Path to the source documents")
    gen_parser.add_argument(
        "output",
        help="Path to save generated QA pairs (JSONL/CSV/etc.)",
    )

    # --- train-retriever ---------------------------------------------------
    train_parser = subparsers.add_parser(
        "train-retriever",
        help="Train a retriever model on QA pairs.",
    )
    train_parser.add_argument("qa_path", help="Path to the QA-pairs dataset")
    train_parser.add_argument(
        "--output-dir",
        default="models/retriever",
        help="Where to save the trained retriever (default: %(default)s)",
    )
    train_parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Backbone model name (default: %(default)s)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Entry point for the console script."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "generate-qa":
        from docs2synth import qa

        docs = qa.load_documents(args.input)
        qa_pairs = qa.generate_qa_pairs(docs)

        # TODO: write to args.output (e.g. JSONL). Placeholder implementation.
        print("Generated QA pairs, but writing is not yet implemented.")

    elif args.command == "train-retriever":
        from docs2synth import retriever

        qa_pairs = retriever.load_qa_pairs(args.qa_path)
        model_dir = retriever.train_retriever(
            qa_pairs,
            model_name=args.model_name,
            output_dir=args.output_dir,
        )
        print(f"Model saved to {model_dir}")

    else:
        # No subcommand provided; show help
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main() 