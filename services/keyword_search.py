from models.movie import Movie
from nltk.stem import PorterStemmer
from services.utils import load_movies, load_stopwords, remove_punc


stemmer = PorterStemmer()
movies = load_movies()
stopwords = load_stopwords()

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

