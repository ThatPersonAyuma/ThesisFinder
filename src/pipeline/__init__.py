"""Search Pipeline Module - Orchestration filtering, VSM, dan BM25F."""

from importlib import import_module

_EXPORTS = {
    "SearchPipeline": ("search_pipeline", "SearchPipeline"),
    "ensure_indexes": ("setup", "ensure_indexes"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        mod = import_module(f".{mod_name}", __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
