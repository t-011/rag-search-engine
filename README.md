# RAG Search Engine

The retrieval layer of a retrieval-augmented generation (RAG) pipeline, written in Python and searching about 5,000 movie descriptions.

Built while working through Boot.dev's RAG course.

## What's implemented

**Keyword search** (`services/inverted_index.py`)
- Inverted index with stop-word removal and Porter stemming (NLTK)
- TF, IDF, TF-IDF, and BM25 ranking (tunable `k1` and `b`), written from scratch
- Index cached to disk with pickle

**Semantic search** (`services/semantic_search.py`)
- Document embeddings from `sentence-transformers` (`all-MiniLM-L6-v2`)
- Results ranked by cosine similarity, embeddings cached as `.npy`
- Fixed-size and sentence-based chunking with overlap, plus chunk embeddings

## In progress

- Chunk-level search (score chunks, aggregate per movie)
- Hybrid ranking (BM25 combined with semantic scores)
- LLM answer generation on top of retrieved results

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put the dataset in `data/`. It isn't committed because it's 25 MB:
- `data/movies.json`, in the format `{"movies": [{"id", "title", "description"}, ...]}`
- `data/stopwords.txt`, with one word per line

## Usage

Run from the repo root.

```bash
# keyword search
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "space adventure" --limit 5
uv run cli/keyword_search_cli.py tfidf 1763 space

# semantic search (builds embeddings on first run)
uv run cli/semantic_search_cli.py search "a robot falls in love" --limit 5
uv run cli/semantic_search_cli.py semantic-chunk "First sentence. Second one. Third." --max-chunk-size 2 --overlap 1
uv run cli/semantic_search_cli.py embed-chunks
```
