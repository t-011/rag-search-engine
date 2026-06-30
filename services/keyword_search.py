import json
import string
from models.movie import Movie
from nltk.stem import PorterStemmer

MOVIES_JSON_FILE = "/Users/home/Documents/Projects/bootdev/rag/rag-search-engine/data/movies.json"
STOPWORDS_TXT_FILE = "/Users/home/Documents/Projects/bootdev/rag/rag-search-engine/data/stopwords.txt"
_REMOVE_PUNC_TABLE = str.maketrans("", "", string.punctuation)



def remove_punc(s: str) -> str:
    return s.translate(_REMOVE_PUNC_TABLE)


def load_movies(file: str = MOVIES_JSON_FILE) -> list[Movie]:
    raw = json.loads(open(file).read())["movies"]
    return [Movie(**m) for m in raw]


def load_stopwords(file: str = STOPWORDS_TXT_FILE) -> set[str]:
    with open(file) as f:
        return {remove_punc(w) for w in f.read().splitlines()}


movies = load_movies(MOVIES_JSON_FILE)
stopwords = load_stopwords(STOPWORDS_TXT_FILE)
stemmer = PorterStemmer()

def tokenize(text: str, stopwords: set[str] = stopwords, stemmer: PorterStemmer = stemmer) -> list[str]:
    tokens = [w for w in remove_punc(text.lower()).split() if w not in stopwords]
    return [stemmer.stem(t) for t in tokens]

def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"expected one token, got {len(tokens)}: {tokens!r}")
    return tokens[0]


def search_by_title(
     title: str, 
     movies: list[Movie] = movies,
) -> list[Movie]:
    query_stems = tokenize(title)

    result: list[Movie] = []
    for movie in movies:
        movie_stems = tokenize(movie.title)
        if set(query_stems) & set(movie_stems):
            result.append(movie)
    return result


def keyword_search(title: str) -> list[Movie]:
    return search_by_title(title)


if __name__ == "__main__":
    pass
