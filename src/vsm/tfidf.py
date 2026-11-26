"""Dasar VSM: konfigurasi, pra-pemrosesan teks, dan rumus TF-IDF (skema "ltc").

Bobot dokumen/query untuk term t:
    tf(t, d)  = 1 + ln(count(t, d))                 jika count > 0, selain itu 0   (sublinear tf)
    idf(t)    = ln((N + 1) / (df(t) + 1)) + 1                                       (idf smooth)
    w(t, d)   = tf(t, d) * idf(t)
    v_d       = w(., d) / ||w(., d)||_2                                             (normalisasi L2)
Kemiripan query-dokumen = cosine = dot(v_q, v_d), karena keduanya sudah unit-length.
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

# ===== 1. KONFIGURASI =====
ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT_DIR / "data" / "completed.parquet"
VSM_DATA_DIR = ROOT_DIR / "data" / "vsm"
VOCAB_PATH = VSM_DATA_DIR / "vocab_clean.csv"
INDEX_DIR = ROOT_DIR / "data" / "index" / "vsm"

FIELDS = ("title", "abstract")
OPTIONAL_FIELDS = ("keywords", "metode", "evaluasi", "kesimpulan")

DEFAULT_FIELD_WEIGHTS = {
    "title": 3.0,
    "keywords": 2.0,
    "abstract": 1.0,
    "metode": 1.0,
    "evaluasi": 1.0,
    "kesimpulan": 1.0,
}

INDEX_FORMAT_VERSION = 2


# ===== 2. PRA-PEMROSESAN TEKS =====
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours yourself yourselves
however thus therefore may might must shall via using use used based within without among upon et al
""".split()
)

STOPWORDS_FREK = frozenset(
    """
across often many due well several even towards toward yet still less become becomes becoming whether
much since although either beyond along thereby rather cannot per around whose every least namely another
together always though almost whereas behind already mostly latter enough hence throughout sometimes
others done made
""".split()
)
STOPWORDS = STOPWORDS | STOPWORDS_FREK


@lru_cache(maxsize=200_000)
def light_stem(token: str) -> str:
    n = len(token)
    if n <= 3 or token.isdigit():
        return token
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("ches", "shes", "xes", "zes")) and n > 4:
        return token[:-2]
    if token.endswith("ies") and n > 4:
        return token[:-3] + "y"
    if token.endswith("ied") and n > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and n - 3 >= 4:
        return token[:-3]
    if token.endswith("ed") and n - 2 >= 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Strip URL, normalisasi ASCII, ambil token alfanumerik >1 karakter (bukan angka murni)."""
    if not text:
        return []
    cleaned = _URL_RE.sub(" ", str(text))
    cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    return [t for t in _TOKEN_RE.findall(cleaned.lower()) if len(t) > 1 and not t.isdigit()]


def analyze_text(text: str) -> list[str]:
    """Tokenisasi + stopword removal + stemming ringan."""
    return [light_stem(t) for t in tokenize(text) if t not in STOPWORDS]


FIELD_ANALYZERS = {
    "title": analyze_text,
    "abstract": analyze_text,
    "keywords": analyze_text,
    "metode": analyze_text,
    "evaluasi": analyze_text,
    "kesimpulan": analyze_text,
}


# ===== 3. RUMUS TF-IDF =====
def term_frequency(counts: np.ndarray) -> np.ndarray:
    """Sublinear TF: 1 + ln(count). Input count harus > 0."""
    return 1.0 + np.log(counts.astype(np.float64))


def inverse_document_frequency(df: np.ndarray, n_docs: int) -> np.ndarray:
    """IDF smooth: ln((N+1)/(df+1)) + 1. Term dengan df == 0 diberi idf 0."""
    df = df.astype(np.float64)
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    idf[df == 0] = 0.0
    return idf


def row_ids_from_indptr(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def l2_normalize_rows(
    data: np.ndarray, indptr: np.ndarray, rows: np.ndarray | None = None
) -> np.ndarray:
    if len(data) == 0:
        return data.astype(np.float64)
    n_rows = len(indptr) - 1
    if rows is None:
        rows = row_ids_from_indptr(indptr)
    sq = np.bincount(rows, weights=data.astype(np.float64) ** 2, minlength=n_rows)
    norms = np.sqrt(sq)
    norms[norms == 0] = 1.0
    return data / norms[rows]


def weight_matrix_data(
    indices: np.ndarray,
    counts: np.ndarray,
    indptr: np.ndarray,
    n_terms: int,
    rows: np.ndarray | None = None,
):
    nonempty = int(np.count_nonzero(np.diff(indptr)))
    df = np.bincount(indices, minlength=n_terms)
    idf = inverse_document_frequency(df, nonempty)
    if len(indices) == 0:
        return np.empty(0, dtype=np.float64), idf, nonempty
    tf = term_frequency(counts)
    w = tf * idf[indices]
    return l2_normalize_rows(w, indptr, rows=rows), idf, nonempty


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
