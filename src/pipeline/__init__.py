"""
Search Pipeline Module - Orchestration filtering dan search.

Modul ini mengintegrasikan BinaryFilter, VSMSearcher, dan metadata enrichment
untuk menyediakan workflow pencarian lengkap.
"""

from .search_pipeline import SearchPipeline

__all__ = ["SearchPipeline"]
