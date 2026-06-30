from collections import Counter
import pickle, math
from pydoc import doc

from services.keyword_search import tokenize, load_movies
from models.movie import Movie
from pydantic import BaseModel
import os

CACHE_DIR = "cache"
INDEX_PKL_FILE = "cache/index.pkl" # perspective of root
DOCMAP_PKL_FILE = "cache/docmap.pkl"
TERMFREQ_PKL_FILE = "cache/term_frequencies.pkl"
DOCLENGTHS_PKL_FILE = "cache/doc_lengths.pkl"

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex(BaseModel):
    index: dict[str, set[int]] = dict()
    docmap: dict[int, Movie] = dict()
    term_freq: dict[int, Counter] = dict()
    doc_lengths: dict[int, int] = dict()
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 1.0
        
        total_doclengths = sum(self.doc_lengths.values())
        docs = len(self.doc_lengths.keys())

        return total_doclengths / docs

    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize(text)

        # index
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

        # term_freq
        self.term_freq[doc_id] = Counter(tokens)

        # doc_length
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index[term]) if term in self.index else []
    
    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_freq.get(doc_id, Counter())[term] 
    
    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap) + 1) / (len(self.get_documents(term)) + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap.keys())
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        doc_length = self.doc_lengths.get(doc_id, 1)
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())

        tf = self.get_tf(doc_id, term)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query, limit):
        tokens = tokenize(query)
        scores: dict[int, float] = dict()

        for doc_id in self.docmap.keys():
            score: float = sum([self.bm25(doc_id, token) for token in tokens])
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores[:limit]


    
    def build(self, docs: list[Movie] = load_movies()):
        for movie in docs:
            self.__add_document(movie.id, f"{movie.title} {movie.description}")
            self.docmap[movie.id] = movie

    def load(self, 
             index_file: str = INDEX_PKL_FILE, 
             docmap_file:str = DOCMAP_PKL_FILE,
             term_freq_file: str = TERMFREQ_PKL_FILE,
             doc_lengths_file: str = DOCLENGTHS_PKL_FILE):
        
        with open(index_file, "rb") as f:
            self.index = pickle.load(f)
        
        with open(docmap_file, "rb") as f:
            self.docmap = pickle.load(f)

        with open(term_freq_file, "rb") as f:
            self.term_freq = pickle.load(f)

        with open(doc_lengths_file, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def save(self, 
             index_file: str = INDEX_PKL_FILE, 
             docmap_file:str = DOCMAP_PKL_FILE,
             term_freq_file: str = TERMFREQ_PKL_FILE,
             doc_lengths_file: str = DOCLENGTHS_PKL_FILE):
        
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)
        
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(term_freq_file, "wb") as f:
            pickle.dump(self.term_freq, f)

        with open(doc_lengths_file, "wb") as f:
            pickle.dump(self.doc_lengths, f)


    




