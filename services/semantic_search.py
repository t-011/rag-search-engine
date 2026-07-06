from dataclasses import dataclass
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import embedding
from models.movie import Movie
import os
from services.utils import load_movies, load_stopwords
from services.hash import NUM_BUCKETS, hash_vec
import re, json

VECTOR_STORE_NPY_FILE = "cache/movie_embeddings.npy"

@dataclass
class SimilarityResult:
    score: float
    title: str
    description: str

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    movies = load_movies()
    embeddings = model.load_or_create_embeddings(movies)
    
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def search(query: str, limit: int):
    model = SemanticSearch()
    movies = load_movies()
    model.load_or_create_embeddings(movies)

    similar_movies: list[SimilarityResult] = model.search(query, limit)
    for i, m in enumerate(similar_movies, start=1):
        print(f"{i}. {m.title} (score: {m.score})")
        print(f"  {m.description}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


    

class SemanticSearch():
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, device="cpu")
        self.embeddings = None
        self.documents = None
        self.embedding_buckets: list[list[tuple]] = [[] for _ in range(NUM_BUCKETS)]
        self.document_map = {}

    def _save_embedding_buckets(self, embeddings: list, movies: list[Movie]):
        if embeddings is None:
            raise ValueError("embeddings must have values to put into buckets")
        
        for emb, movie in zip(embeddings, movies):
            self.embedding_buckets[hash_vec(emb)].append((emb, movie))


    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("text must not be empty")
        
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents: List[Movie]):
        self.documents = documents
        doc_strings: List[str] = []

        for doc in documents:
            self.document_map[doc.id] = doc
            doc_strings.append(f"{doc.title}: {doc.description}")

        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        self._save_embedding_buckets(self.embeddings, documents)

        np.save(VECTOR_STORE_NPY_FILE, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents: List[Movie]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc

        if os.path.exists(VECTOR_STORE_NPY_FILE):
            self.embeddings = np.load(VECTOR_STORE_NPY_FILE)
            if len(self.embeddings) == len(documents):
                self._save_embedding_buckets(self.embeddings, documents)
                return self.embeddings
            
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int) -> list[SimilarityResult]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_vec = self.generate_embedding(query)
        cos_similarities: list[tuple[float, Movie]] = []

        for doc_vec, doc in self.embedding_buckets[hash_vec(query_vec)]:
            similarity = cosine_similarity(doc_vec, query_vec)
            cos_similarities.append((similarity, doc))

        most_similar_movies = sorted(cos_similarities, key=lambda x: x[0], reverse=True)[:limit]

        top_results: list[SimilarityResult] = []
        for score, movie in most_similar_movies:
            top_results.append(SimilarityResult(score, movie.title, movie.description))

        return top_results

CHUNK_EMBEDDINGS_CACHE_NPY = "cache/chunk_embeddings.npy"
CHUNK_METADATA_JSON = "cache/chunk_metadata.json"

def semantic_chunking(text: str, max_chunk_size: int = 4, overlap: int = 0):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks: list[str] = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i: i+max_chunk_size])
        chunks.append(chunk)
    
    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in documents: 
            self.document_map[doc.id] = doc

        chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for doc_idx, doc in enumerate(documents):
            if not doc.description: continue

            curr_chunks = semantic_chunking(doc.description, max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(curr_chunks):
                chunks.append(chunk)
                chunk_metadata.append({"movie_idx": doc_idx, "chunk_idx": chunk_idx, "total_chunks": len(curr_chunks)})

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = chunk_metadata

        np.save(CHUNK_EMBEDDINGS_CACHE_NPY, self.chunk_embeddings)
        with open(CHUNK_METADATA_JSON, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        
        return np.asarray(self.chunk_embeddings)
    
    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in documents: 
            self.document_map[doc.id] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_CACHE_NPY) and os.path.exists(CHUNK_METADATA_JSON):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_CACHE_NPY)
            with open(CHUNK_METADATA_JSON, "r") as f:
                self.chunk_metadata = json.load(f)
            
            return np.asarray(self.chunk_embeddings)
        
        return self.build_chunk_embeddings(documents)
    




            


        







