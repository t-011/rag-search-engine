from models.movie import Movie
import json
import string


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
