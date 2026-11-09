from models.movie import Movie
import json
import string
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIES_JSON_FILE = DATA_DIR / "movies.json"
STOPWORDS_TXT_FILE = DATA_DIR / "stopwords.txt"
_REMOVE_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def remove_punc(s: str) -> str:
    return s.translate(_REMOVE_PUNC_TABLE)

def load_movies(file: str | Path = MOVIES_JSON_FILE) -> list[Movie]:
    with open(file) as f:
        raw = json.load(f)["movies"]
    return [Movie(**m) for m in raw]


def load_stopwords(file: str | Path = STOPWORDS_TXT_FILE) -> set[str]:
    with open(file) as f:
        return {remove_punc(w) for w in f.read().splitlines()}
