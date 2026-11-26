"""Modul VSM (Vector Space Model) berbasis TF-IDF dan cosine similarity."""

from importlib import import_module

_EXPORTS = {
    "VSMSearcher": ("searcher", "VSMSearcher"),
    "SearchResult": ("searcher", "SearchResult"),
    "Tokenizer": ("cleaning.tokenizer", "Tokenizer"),
    "build_index": ("indexer", "build_index"),
    "build_index_from_parquet": ("indexer", "build_index_from_parquet"),
    "iter_papers": ("indexer", "iter_papers"),
    "analyze_text": ("tfidf", "analyze_text"),
    "light_stem": ("tfidf", "light_stem"),
    "tokenize": ("tfidf", "tokenize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        mod = import_module(f".{mod_name}", __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
