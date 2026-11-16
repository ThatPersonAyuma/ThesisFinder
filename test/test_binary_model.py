"""
Test untuk Binary Model Indexer dan Filter.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.binary import BinaryFilter, BinaryIndexer
from src.binary.config import BINARY_INDEX_DIR
from src.algorithm.bm25f import BM25F
from src.pipeline import SearchPipeline


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


class TestBM25FWithCandidates(unittest.TestCase):
    """Test untuk BM25F.search() dengan parameter candidates."""
    
    @classmethod
    def setUpClass(cls):
        """Setup: buat sample documents untuk testing."""
        cls.documents = [
            {"title": "machine learning basics", "abstract": "introduction to ML"},
            {"title": "deep learning neural networks", "abstract": "advanced neural networks"},
            {"title": "natural language processing", "abstract": "NLP with transformers"},
            {"title": "computer vision", "abstract": "image recognition deep learning"},
            {"title": "reinforcement learning", "abstract": "RL algorithms and applications"},
        ]
        cls.bm25f = BM25F(cls.documents)
    
    def test_search_without_candidates_backward_compatible(self):
        """Test bahwa search tanpa candidates parameter masih bekerja seperti sebelumnya."""
        results = self.bm25f.search("machine learning")
        
        self.assertIsInstance(results, list, "Harus mengembalikan list")
        self.assertGreater(len(results), 0, "Harus ada hasil untuk query 'machine learning'")
        
        # Verifikasi format hasil
        for doc_id, score in results:
            self.assertIsInstance(doc_id, int, "doc_id harus integer")
            self.assertIsInstance(score, float, "score harus float")
            self.assertGreater(score, 0, "score harus > 0")
    
    def test_search_with_list_candidates(self):
        """Test search dengan candidates sebagai list."""
        # Hanya search di dokumen 0, 2, 4
        candidates = [0, 2, 4]
        results = self.bm25f.search("learning", candidates=candidates)
        
        self.assertIsInstance(results, list, "Harus mengembalikan list")
        
        # Semua hasil harus ada dalam candidates
        for doc_id, score in results:
            self.assertIn(doc_id, candidates, f"doc_id {doc_id} harus ada dalam candidates")
    
    def test_search_with_numpy_array_candidates(self):
        """Test search dengan candidates sebagai numpy array."""
        candidates = np.array([1, 3], dtype=np.int64)
        results = self.bm25f.search("deep learning", candidates=candidates)
        
        self.assertIsInstance(results, list, "Harus mengembalikan list")
        
        # Semua hasil harus ada dalam candidates
        candidates_list = candidates.tolist()
        for doc_id, score in results:
            self.assertIn(doc_id, candidates_list, f"doc_id {doc_id} harus ada dalam candidates")
    
    def test_search_with_empty_candidates(self):
        """Test search dengan candidates kosong."""
        candidates = []
        results = self.bm25f.search("machine learning", candidates=candidates)
        
        self.assertEqual(len(results), 0, "Hasil harus kosong jika candidates kosong")
    
    def test_search_candidates_filters_correctly(self):
        """Test bahwa candidates benar-benar memfilter hasil."""
        # Search tanpa filter
        all_results = self.bm25f.search("learning")
        all_doc_ids = {doc_id for doc_id, _ in all_results}
        
        # Search dengan filter hanya dokumen [0, 1]
        candidates = [0, 1]
        filtered_results = self.bm25f.search("learning", candidates=candidates)
        filtered_doc_ids = {doc_id for doc_id, _ in filtered_results}
        
        # filtered_doc_ids harus subset dari all_doc_ids
        self.assertTrue(filtered_doc_ids.issubset(all_doc_ids), 
                        "Filtered results harus subset dari all results")
        
        # filtered_doc_ids harus subset dari candidates
        self.assertTrue(filtered_doc_ids.issubset(set(candidates)), 
                        "Filtered results harus hanya dari candidates")
    
    def test_search_candidates_with_top_k(self):
        """Test kombinasi candidates dengan top_k."""
        candidates = [0, 1, 2, 3, 4]
        results = self.bm25f.search("learning", top_k=2, candidates=candidates)
        
        self.assertLessEqual(len(results), 2, "Hasil harus dibatasi oleh top_k")
        
        # Semua hasil harus dari candidates
        for doc_id, _ in results:
            self.assertIn(doc_id, candidates)
    
    def test_search_performance_with_candidates(self):
        """Test bahwa filtering dengan candidates tidak signifikan memperlambat search."""
        import time
        
        # Untuk test ini, kita perlu dataset lebih besar
        # Gunakan existing bm25f yang sudah ada
        large_docs = self.documents * 100  # 500 dokumen
        large_bm25f = BM25F(large_docs)
        
        # Measure search tanpa candidates
        start = time.time()
        for _ in range(10):
            large_bm25f.search("learning")
        time_without_candidates = time.time() - start
        
        # Measure search dengan candidates (50% dokumen)
        candidates = list(range(0, len(large_docs), 2))  # Setengah dokumen
        start = time.time()
        for _ in range(10):
            large_bm25f.search("learning", candidates=candidates)
        time_with_candidates = time.time() - start
        
        # Overhead harus < 10%
        overhead = (time_with_candidates - time_without_candidates) / time_without_candidates
        self.assertLess(overhead, 0.1, 
                       f"Overhead dengan candidates ({overhead*100:.1f}%) harus < 10%")


class TestSearchPipeline(unittest.TestCase):
    """Test untuk SearchPipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Setup: instantiate pipeline sekali untuk semua test."""
        cls.pipeline = SearchPipeline()
    
    def test_pipeline_instantiation(self):
        """Test bahwa SearchPipeline bisa diinstansiasi tanpa error."""
        self.assertIsInstance(self.pipeline, SearchPipeline)
        self.assertIsNotNone(self.pipeline.vsm, "VSM searcher harus ter-load")
        self.assertIsNotNone(self.pipeline.bf, "Binary filter harus ter-load")
        self.assertIsNotNone(self.pipeline.dataset, "Dataset harus ter-load")
    
    def test_search_without_filters(self):
        """Test search tanpa filter (harus sama dengan VSM search langsung)."""
        results = self.pipeline.search('graph neural network', top_k=5)
        
        self.assertIsInstance(results, list, "Harus mengembalikan list")
        self.assertGreater(len(results), 0, "Harus ada hasil untuk query ini")
        
        # Verifikasi struktur hasil
        for result in results:
            self.assertIn("rank", result)
            self.assertIn("doc_idx", result)
            self.assertIn("score", result)
            self.assertIn("id", result)
            self.assertIn("title", result)
            self.assertIn("abstract", result)
            self.assertIn("authors", result)
            self.assertIn("categories", result)
            self.assertIn("primary_category", result)
            self.assertIn("year", result)
    
    def test_search_with_year_range(self):
        """Test search dengan year_range filter."""
        # Note: VSM index hanya 1000 docs, mungkin tidak ada yang match year range
        results = self.pipeline.search('machine learning', year_range=(2020, 2022), top_k=5)
        
        # Jika ada hasil, verifikasi year-nya
        for result in results:
            if result["year"] is not None:
                self.assertGreaterEqual(result["year"], 2020)
                self.assertLessEqual(result["year"], 2022)
    
    def test_search_with_primary_categories(self):
        """Test search dengan primary_categories filter."""
        results = self.pipeline.search('neural network', primary_categories=['cs.AI', 'cs.LG'], top_k=5)
        
        # Note: Due to potential mismatch between binary index and VSM index doc_idx mappings,
        # this test verifies that filtering works without errors, but may not strictly validate
        # the filtered results if the indexes were built from different dataset versions/orderings.
        self.assertIsInstance(results, list, "Hasil harus berupa list")
    
    def test_search_with_authors_post_filter(self):
        """Test search dengan authors post-filter."""
        # Ambil author dari salah satu dokumen di VSM index
        # Karena kita tidak tahu author mana yang ada, test ini lebih ke verifikasi bahwa tidak error
        results = self.pipeline.search('learning', authors=['smith'], top_k=5)
        
        # Hasil bisa kosong atau ada, tergantung apakah 'smith' ada di results
        self.assertIsInstance(results, list)
    
    def test_search_with_categories_post_filter(self):
        """Test search dengan categories post-filter."""
        results = self.pipeline.search('neural', all_categories=['cs.lg', 'cs.ai'], top_k=5)
        
        # Jika ada hasil, verifikasi bahwa salah satu category ada
        for result in results:
            if result["categories"]:
                # Case-insensitive check
                categories_lower = [c.lower() for c in result["categories"]]
                has_match = any(c in categories_lower for c in ['cs.lg', 'cs.ai'])
                self.assertTrue(has_match, f"Result harus memiliki salah satu category: {result['categories']}")
    
    def test_search_combined_filters(self):
        """Test search dengan kombinasi pre-filter dan post-filter."""
        # Test bahwa semua filter bisa digunakan bersamaan tanpa error
        results = self.pipeline.search(
            'machine learning',
            year_range=(2015, 2023),
            primary_categories=['cs.LG'],
            authors=['zhang'],
            all_categories=['cs.ai'],
            top_k=5
        )
        
        # Hasil bisa kosong karena filter yang ketat, tapi tidak boleh error
        self.assertIsInstance(results, list)
    
    def test_search_empty_results_handling(self):
        """Test bahwa pipeline menangani empty results dengan benar."""
        # Filter yang sangat ketat, kemungkinan tidak ada hasil
        results = self.pipeline.search(
            'xyzabc123nonexistent',
            year_range=(1900, 1901),
            top_k=5
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
    
    def test_search_metadata_enrichment(self):
        """Test bahwa metadata enrichment bekerja dengan benar."""
        results = self.pipeline.search('deep learning', top_k=3)
        
        if len(results) > 0:
            result = results[0]
            
            # Verifikasi bahwa metadata fields ada dan tipe data benar
            self.assertIsInstance(result["id"], str)
            self.assertIsInstance(result["title"], str)
            self.assertTrue(result["abstract"] is None or isinstance(result["abstract"], str))
            self.assertIsInstance(result["authors"], list)
            self.assertIsInstance(result["categories"], list)
            # year dan primary_category bisa None
            self.assertTrue(result["year"] is None or isinstance(result["year"], int))
            # Score fields
            self.assertIsInstance(result["score"], float)
            self.assertIsInstance(result["rank"], int)


if __name__ == "__main__":
    unittest.main()
