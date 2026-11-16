"""
Binary Model Module - Inverted index dan filtering untuk metadata fields.

Modul ini menyediakan:
- BinaryIndexer: Membangun inverted index untuk year, primary_category, authors, categories
- BinaryFilter: Pre-filtering dan post-filtering berdasarkan metadata
"""

from .filter import BinaryFilter
from .indexer import BinaryIndexer

__all__ = ["BinaryFilter", "BinaryIndexer"]
