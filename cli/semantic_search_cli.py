import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from services.semantic_search import embed_query_text, search, verify_model, embed_text, verify_embeddings

def verify_command():
    verify_model()

def embed_text_command(text: str):
    embed_text(text)

def verify_embeddings_command():
    verify_embeddings()

def embed_query_command(query: str):
    embed_query_text(query)

def search_command(query: str, limit: int):
    search(query, limit)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_subparser = subparsers.add_parser("verify")

    embed_subparser = subparsers.add_parser("embed")
    embed_subparser.add_argument("text", type=str)

    verify_embeddings_subparser = subparsers.add_parser("verify-embeddings")

    embed_query_text_subparser = subparsers.add_parser("embed-query-text") 
    embed_query_text_subparser.add_argument("query", type=str)

    search_subparser = subparsers.add_parser("search")
    search_subparser.add_argument("query", type=str)
    search_subparser.add_argument("--limit", type=int, default=5)


    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_command()
        
        case "embed":
            embed_text_command(args.text)

        case "verify-embeddings":
            verify_embeddings_command()

        case "embed-query-text":
            embed_query_command(args.query)
        
        case "search":
            search_command(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()