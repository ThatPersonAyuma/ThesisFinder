"""
Test untuk Binary Model Indexer dan Filter.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.binary import BinaryFilter, BinaryIndexer
from src.binary.config import BINARY_INDEX_DIR


class TestBinaryIndexer(unittest.TestCase):
    """Test untuk BinaryIndexer."""
    
    def test_indexes_exist(self):
        """Verifikasi bahwa semua file index ada."""
        index_dir = Path(BINARY_INDEX_DIR)
        self.assertTrue((index_dir / "config.json").exists(), "config.json harus ada")
        self.assertTrue((index_dir / "year.parquet").exists(), "year.parquet harus ada")
        self.assertTrue((index_dir / "primary_category.parquet").exists(), "primary_category.parquet harus ada")
        self.assertTrue((index_dir / "authors.parquet").exists(), "authors.parquet harus ada")
        self.assertTrue((index_dir / "categories.parquet").exists(), "categories.parquet harus ada")
    
    def test_index_structure(self):
        """Verifikasi struktur index parquet files."""
        index_dir = Path(BINARY_INDEX_DIR)
        
        for filename in ["year.parquet", "primary_category.parquet", "authors.parquet", "categories.parquet"]:
            df = pd.read_parquet(index_dir / filename)
            self.assertIn("key", df.columns, f"{filename} harus memiliki kolom 'key'")
            self.assertIn("doc_idx", df.columns, f"{filename} harus memiliki kolom 'doc_idx'")
    
    def test_authors_multiple_entries(self):
        """Verifikasi bahwa multi-author papers memiliki multiple entries."""
        index_dir = Path(BINARY_INDEX_DIR)
        authors_df = pd.read_parquet(index_dir / "authors.parquet")
        
        # Cek apakah ada dokumen dengan multiple authors
        doc_counts = authors_df["doc_idx"].value_counts()
        multi_author_docs = (doc_counts > 1).sum()
        
        self.assertGreater(multi_author_docs, 0, "Harus ada dokumen dengan multiple authors")


class TestBinaryFilter(unittest.TestCase):
    """Test untuk BinaryFilter."""
    
    @classmethod
    def setUpClass(cls):
        """Setup: load filter sekali untuk semua test."""
        cls.filter = BinaryFilter()
    
    def test_filter_instantiation(self):
        """Test bahwa BinaryFilter bisa diinstansiasi tanpa error."""
        self.assertIsInstance(self.filter, BinaryFilter)
    
    def test_filter_by_year(self):
        """Test filtering berdasarkan year range."""
        candidates = self.filter.filter_by_year(2020, 2022)
        
        self.assertIsInstance(candidates, np.ndarray, "Harus mengembalikan numpy array")
        self.assertGreater(len(candidates), 0, "Harus ada kandidat untuk year 2020-2022")
        self.assertEqual(candidates.dtype, np.int64, "Doc indices harus bertipe int64")
    
    def test_filter_by_primary_category(self):
        """Test filtering berdasarkan primary_category."""
        candidates = self.filter.filter_by_primary_category(["cs.AI", "cs.LG"])
        
        self.assertIsInstance(candidates, np.ndarray, "Harus mengembalikan numpy array")
        self.assertGreater(len(candidates), 0, "Harus ada kandidat untuk cs.AI atau cs.LG")
    
    def test_get_pre_filter_candidates_intersection(self):
        """Test kombinasi pre-filtering dengan AND logic (intersection)."""
        # Filter dengan year range dan primary_category
        candidates = self.filter.get_pre_filter_candidates(
            year_range=(2020, 2022),
            primary_categories=["cs.AI", "cs.LG"]
        )
        
        self.assertIsInstance(candidates, np.ndarray, "Harus mengembalikan numpy array")
        self.assertGreater(len(candidates), 0, "Harus ada kandidat yang memenuhi kedua filter")
        
        # Verifikasi bahwa hasilnya adalah intersection (lebih sedikit dari filter individual)
        year_only = self.filter.get_pre_filter_candidates(year_range=(2020, 2022))
        cat_only = self.filter.get_pre_filter_candidates(primary_categories=["cs.AI", "cs.LG"])
        
        self.assertLessEqual(len(candidates), len(year_only), "Intersection harus <= year filter")
        self.assertLessEqual(len(candidates), len(cat_only), "Intersection harus <= category filter")
    
    def test_filter_by_authors_case_insensitive(self):
        """Test post-filtering by authors dengan case-insensitive matching."""
        # Ambil sample candidates
        sample_candidates = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        
        # Filter dengan author name (test case-insensitive)
        # Kita perlu tahu nama author yang valid dulu
        authors_df = pd.read_parquet(self.filter.index_dir / "authors.parquet")
        sample_author = authors_df.iloc[0]["key"]
        
        # Test dengan berbagai case
        result_lower = self.filter.filter_by_authors(sample_candidates, [sample_author.lower()])
        result_upper = self.filter.filter_by_authors(sample_candidates, [sample_author.upper()])
        
        # Kedua hasil harus sama (case-insensitive)
        self.assertEqual(len(result_lower), len(result_upper), "Case-insensitive matching harus konsisten")
    
    def test_filter_by_categories_any_logic(self):
        """Test post-filtering by categories dengan ANY logic."""
        # Ambil sample candidates
        sample_candidates = self.filter.filter_by_year(2020, 2022)[:100]
        
        # Filter dengan multiple categories (ANY logic: OR)
        result = self.filter.filter_by_categories(sample_candidates, ["cs.ai", "cs.lg"])
        
        self.assertIsInstance(result, np.ndarray, "Harus mengembalikan numpy array")
        # Result bisa kosong atau ada, tergantung data
    
    def test_apply_post_filters(self):
        """Test kombinasi post-filtering dengan AND logic."""
        # Ambil pre-filter candidates
        pre_candidates = self.filter.get_pre_filter_candidates(year_range=(2020, 2022))
        
        # Ambil sample authors dan categories dari index
        authors_df = pd.read_parquet(self.filter.index_dir / "authors.parquet")
        categories_df = pd.read_parquet(self.filter.index_dir / "categories.parquet")
        
        sample_author = authors_df.iloc[0]["key"]
        sample_category = categories_df.iloc[0]["key"]
        
        # Apply post-filters
        result = self.filter.apply_post_filters(
            pre_candidates[:100],
            authors=[sample_author],
            categories=[sample_category]
        )
        
        self.assertIsInstance(result, np.ndarray, "Harus mengembalikan numpy array")


if __name__ == "__main__":
    unittest.main()
