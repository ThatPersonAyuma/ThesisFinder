"""Submodul pembersihan kosakata, stopword, stemming, dan tokenisasi VSM."""

from importlib import import_module

_EXPORTS = {
    "Tokenizer": ("tokenizer", "Tokenizer"),
    "build_vocab_csv": ("tokenizer", "build_vocab_csv"),
    "build_word_freq_from_parquet": ("cleaning", "build_word_freq_from_parquet"),
    "clean_frequencies": ("cleaning", "clean_frequencies"),
    "hapus_stopword": ("stopword", "hapus_stopword"),
    "run_cleaning_pipeline": ("pipeline", "run_cleaning_pipeline"),
    "stemming": ("stemming", "stemming"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        mod = import_module(f".{mod_name}", __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
