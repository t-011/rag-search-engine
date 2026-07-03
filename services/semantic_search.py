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

        
        

    

        







