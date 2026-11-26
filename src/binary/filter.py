"""
Binary Model Filter - Filtering dokumen berdasarkan metadata.

Pre-filtering (year, primary_category): mengembalikan np.array of doc indices
Post-filtering (authors, categories): filtering doc_ids yang sudah ada dengan exact match case-insensitive
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BINARY_INDEX_DIR, DATASET_PATH


class BinaryFilter:
    """
    Filter untuk binary model yang mendukung pre-filtering dan post-filtering.
    
    Pre-filtering: year range dan primary_category (OR logic) -> mengembalikan doc indices
    Post-filtering: authors dan categories (ANY logic dengan case-insensitive exact match) -> filter doc_ids
    """
    
    def __init__(self, index_dir=BINARY_INDEX_DIR, dataset_path=DATASET_PATH):
        """
        Inisialisasi filter dengan memuat semua indeks.
        
        Args:
            index_dir: Direktori yang berisi binary indexes
            dataset_path: Path ke dataset parquet (untuk post-filtering)
        """
        self.index_dir = Path(index_dir)
        self.dataset_path = Path(dataset_path)
        
        # Validasi index directory
        if not (self.index_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Binary indexes belum ada di {self.index_dir}. "
                f"Jalankan: python -m src.binary.indexer"
            )
        
        # Load indexes
        self.year_index = pd.read_parquet(self.index_dir / "year.parquet")
        self.primary_category_index = pd.read_parquet(self.index_dir / "primary_category.parquet")
        self.authors_index = pd.read_parquet(self.index_dir / "authors.parquet")
        self.categories_index = pd.read_parquet(self.index_dir / "categories.parquet")
        
        # Load dataset untuk post-filtering (lazy load saat dibutuhkan)
        self._dataset = None
    
    @property
    def dataset(self):
        """Lazy load dataset."""
        if self._dataset is None:
            self._dataset = pd.read_parquet(self.dataset_path)
        return self._dataset
    
    def filter_by_year(self, year_min: float, year_max: float) -> np.ndarray:
        """
        Filter dokumen berdasarkan range tahun.
        
        Args:
            year_min: Tahun minimum (inclusive)
            year_max: Tahun maksimum (inclusive)
        
        Returns:
            np.array of doc indices yang memenuhi range tahun
        """
        mask = (self.year_index["key"] >= year_min) & (self.year_index["key"] <= year_max)
        doc_indices = self.year_index.loc[mask, "doc_idx"].unique()
        return doc_indices.astype(np.int64)
    
    def filter_by_primary_category(self, categories: list) -> np.ndarray:
        """
        Filter dokumen berdasarkan primary_category dengan OR logic.
        
        Args:
            categories: List of primary_category values (case-sensitive)
        
        Returns:
            np.array of doc indices yang memiliki salah satu primary_category
        """
        if not categories:
            return np.array([], dtype=np.int64)
        
        mask = self.primary_category_index["key"].isin(categories)
        doc_indices = self.primary_category_index.loc[mask, "doc_idx"].unique()
        return doc_indices.astype(np.int64)
    
    def get_pre_filter_candidates(
        self, 
        year_range: tuple = None, 
        primary_categories: list = None
    ) -> np.ndarray:
        """
        Kombinasi pre-filtering dengan AND logic (intersection).
        
        Args:
            year_range: Tuple (year_min, year_max) atau None
            primary_categories: List of primary_category values atau None
        
        Returns:
            np.array of doc indices yang memenuhi semua filter (intersection)
        """
        candidates = None
        
        # Filter by year range
        if year_range is not None:
            year_min, year_max = year_range
            year_candidates = self.filter_by_year(year_min, year_max)
            candidates = year_candidates if candidates is None else np.intersect1d(candidates, year_candidates)
        
        # Filter by primary_category (OR logic)
        if primary_categories is not None:
            cat_candidates = self.filter_by_primary_category(primary_categories)
            candidates = cat_candidates if candidates is None else np.intersect1d(candidates, cat_candidates)
        
        # Jika tidak ada filter, return empty array (tidak ada kandidat)
        if candidates is None:
            return np.array([], dtype=np.int64)
        
        return candidates
    
    def filter_by_authors(self, doc_ids: np.ndarray, authors: list) -> np.ndarray:
        """
        Post-filtering berdasarkan authors dengan ANY logic dan case-insensitive exact match.
        
        Args:
            doc_ids: Array of document indices untuk difilter
            authors: List of author names (case-insensitive)
        
        Returns:
            np.array of doc indices yang memiliki salah satu author
        """
        if not authors or len(doc_ids) == 0:
            return doc_ids
        
        # Normalize authors: lowercase
        authors_normalized = [str(author).lower().strip() for author in authors]
        
        # Filter index berdasarkan normalized authors dan doc_ids
        mask = (
            self.authors_index["key"].isin(authors_normalized) &
            self.authors_index["doc_idx"].isin(doc_ids)
        )
        
        filtered_doc_indices = self.authors_index.loc[mask, "doc_idx"].unique()
        return filtered_doc_indices.astype(np.int64)
    
    def filter_by_categories(self, doc_ids: np.ndarray, categories: list) -> np.ndarray:
        """
        Post-filtering berdasarkan categories dengan ANY logic dan case-insensitive exact match.
        
        Args:
            doc_ids: Array of document indices untuk difilter
            categories: List of category names (case-insensitive)
        
        Returns:
            np.array of doc indices yang memiliki salah satu category
        """
        if not categories or len(doc_ids) == 0:
            return doc_ids
        
        # Normalize categories: lowercase
        categories_normalized = [str(cat).lower().strip() for cat in categories]
        
        # Filter index berdasarkan normalized categories dan doc_ids
        mask = (
            self.categories_index["key"].isin(categories_normalized) &
            self.categories_index["doc_idx"].isin(doc_ids)
        )
        
        filtered_doc_indices = self.categories_index.loc[mask, "doc_idx"].unique()
        return filtered_doc_indices.astype(np.int64)
    
    def apply_post_filters(
        self,
        doc_ids: np.ndarray,
        authors: list = None,
        categories: list = None
    ) -> np.ndarray:
        """
        Aplikasikan post-filtering dengan AND logic (intersection).
        
        Args:
            doc_ids: Array of document indices untuk difilter
            authors: List of author names atau None
            categories: List of category names atau None
        
        Returns:
            np.array of doc indices yang memenuhi semua post-filter
        """
        result = doc_ids.copy()
        
        # Filter by authors
        if authors is not None:
            result = self.filter_by_authors(result, authors)
        
        # Filter by categories
        if categories is not None:
            result = self.filter_by_categories(result, categories)
        
        return result
    
    def __repr__(self):
        return (
            f"BinaryFilter(year={len(self.year_index):,} entries, "
            f"primary_category={len(self.primary_category_index):,} entries, "
            f"authors={len(self.authors_index):,} entries, "
            f"categories={len(self.categories_index):,} entries)"
        )
