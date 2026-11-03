import re
from functools import lru_cache
from pathlib import Path

import numpy as np

# KONFIGURASI 
ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT_DIR / "data" / "papers_2015_2025_keywords.parquet"
INDEX_DIR = ROOT_DIR / "data" / "index" / "vsm"

FIELDS = ("title", "abstract", "keywords", "authors")

# Saat pencarian bobot dinormalisasi terhadap field yang dipilih, sehingga skor akhir berada di [0, 1].
DEFAULT_FIELD_WEIGHTS = {
    "title": 3.0,
    "keywords": 2.0,
    "abstract": 1.0,
    "authors": 1.0,
}

INDEX_FORMAT_VERSION = 1


#  2. PRA-PEMROSESAN TEKS 
_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours yourself yourselves
however thus therefore may might must shall via using use used based within without among upon et al
""".split())


@lru_cache(maxsize=200_000)
def light_stem(token: str) -> str:
    n = len(token)
    if n <= 3 or token.isdigit():
        return token
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ies") and n > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and n - 3 >= 4:
        return token[:-3]
    if token.endswith("ed") and n - 2 >= 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def analyze_text(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def analyze_plain(text: str) -> list[str]:
    return tokenize(text)


FIELD_ANALYZERS = {
    "title": analyze_text,
    "abstract": analyze_text,
    "keywords": analyze_text,
    "authors": analyze_plain,
}


# RUMUS TF-IDF 
def term_frequency(counts: np.ndarray) -> np.ndarray:
    return 1.0 + np.log(counts.astype(np.float64))


def inverse_document_frequency(df: np.ndarray, n_docs: int) -> np.ndarray:
    df = df.astype(np.float64)
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    idf[df == 0] = 0.0
    return idf


def row_ids_from_indptr(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def l2_normalize_rows(data: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    n_rows = len(indptr) - 1
    rows = row_ids_from_indptr(indptr)
    sq = np.bincount(rows, weights=data.astype(np.float64) ** 2, minlength=n_rows)
    norms = np.sqrt(sq)
    norms[norms == 0] = 1.0
    return data / norms[rows]


def weight_matrix_data(indices: np.ndarray, counts: np.ndarray, indptr: np.ndarray,  n_terms: int):

    n_rows = len(indptr) - 1
    nonempty = int(np.count_nonzero(np.diff(indptr)))
    df = np.bincount(indices, minlength=n_terms)
    idf = inverse_document_frequency(df, nonempty)
    tf = term_frequency(counts)
    w = tf * idf[indices]
    return l2_normalize_rows(w, indptr), idf, nonempty


def query_vector(term_counts: dict, idf: np.ndarray):
    if not term_counts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    ids = np.fromiter(term_counts.keys(), dtype=np.int64, count=len(term_counts))
    cnt = np.fromiter(term_counts.values(), dtype=np.float64, count=len(term_counts))
    w = term_frequency(cnt) * idf[ids]
    keep = w > 0
    ids, w = ids[keep], w[keep]
    norm = np.linalg.norm(w)
    if norm == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return ids, w / norm