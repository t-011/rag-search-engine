import re
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from services.semantic_search import ChunkedSemanticSearch, embed_query_text, search, semantic_chunking, verify_model, embed_text, verify_embeddings
from models.movie import Movie
from services.utils import load_movies

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

def chunk_command(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    text_words = text.split()

    chunks: list[str] = []
    step = chunk_size - overlap
    if step <= 0: return []
    for i in range(0, len(text_words), step):
        chunk = " ".join(text_words[i: i+chunk_size])
        chunks.append(chunk)

    return chunks

def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    
    return chunks

def embed_chunks_command(movie_docs: list[Movie] = load_movies()):
    model = ChunkedSemanticSearch()
    embeddings = model.load_or_create_chunk_embeddings(movie_docs)
    print(f"Generated {len(embeddings)} chunked embeddings")

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

    # chunking
    chunk_subparser = subparsers.add_parser("chunk")
    chunk_subparser.add_argument("text", type=str)
    chunk_subparser.add_argument("--chunk-size", type=int, default=200)
    chunk_subparser.add_argument("--overlap", type=int, default=0)

    semantic_chunk_subparser = subparsers.add_parser("semantic-chunk")
    semantic_chunk_subparser.add_argument("text", type=str)
    semantic_chunk_subparser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_subparser.add_argument("--overlap", type=int, default=0)

    embed_chunks_subparser = subparsers.add_parser("embed-chunks")

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

        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size, args.overlap)

            print(f"Chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, start=1):
                print(f"{i}. {c}")

        case "semantic-chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, start=1):
                print(f"{i}. {c}")

        case "embed-chunks":
            embed_chunks_command()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()