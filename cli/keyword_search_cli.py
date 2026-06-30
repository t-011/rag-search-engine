import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from models.movie import Movie
from services.keyword_search import keyword_search, tokenize, tokenize_term

from services.inverted_index import BM25_B, BM25_K1, InvertedIndex

def load_index() -> InvertedIndex:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"File Not Found, error: {e}")
        sys.exit(1)

    return idx

def search_command(args):
    idx = load_index()
    
    query: str = args.query
    seen = set()

    for token in tokenize(query):
        docs = idx.get_documents(token)
        for id in docs:
            if len(seen) < 5 and id not in seen:
                seen.add(id)
                print(f"{idx.docmap[id].title} {idx.docmap[id].id}")


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(args):
    idx = load_index()

    term = tokenize_term(args.term)
    print(idx.get_tf(args.doc_id, term))

def idf_command(args):
    idx = load_index()
    
    term = tokenize_term(args.term)
    idf = idx.get_idf(term)

    print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

def tfidf_command(args):
    idx = load_index()
    term = tokenize_term(args.term)

    tf = idx.get_tf(args.doc_id, term)
    idf = idx.get_idf(term)

    tf_idf = tf * idf

    print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term) -> float:
    idx = load_index()
    term = tokenize_term(term)

    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
    idx = load_index()
    term = tokenize_term(term)

    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_search_command(query: str, limit: int = 5) -> list[tuple[Movie, float]]:
    # returns [(Movie, score))]
    idx = load_index()

    scores: list[tuple[Movie, float]] = []
    scores = [(idx.docmap[doc_id], score) for doc_id, score in idx.bm25_search(query, limit)]

    return scores
    


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index of Movies")

    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get freq of")

    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args)
        
        case "build":
            build_command()

        case "tf":
            tf_command(args)

        case "idf":
            idf_command(args)

        case "tfidf":
            tfidf_command(args)

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            scores = bm25_search_command(args.query, args.limit)
            for movie, score in scores:
                print(f"({movie.id}) {movie.title} - Score: {score:.2f}")
            
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()