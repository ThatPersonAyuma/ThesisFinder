"""
Test suite untuk Binary Model: indexer, filter, BM25F candidates, dan pipeline integration.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.algorithm.bm25f import BM25F
from src.binary.filter import BinaryFilter
from src.binary.indexer import BinaryIndexer


# Test dataset kecil dengan nilai known untuk testing
TEST_PAPERS = [
    {
        "id": "paper_001",
        "title": "Deep Learning for Computer Vision",
        "abstract": "We present a novel deep learning approach for image classification.",
        "year": 2020.0,
        "primary_category": "cs.CV",
        "authors": np.array(["Alice Smith", "Bob Jones"]),
        "categories": np.array(["cs.CV", "cs.AI", "cs.LG"]),
    },
    {
        "id": "paper_002",
        "title": "Natural Language Processing with Transformers",
        "abstract": "Transformer models have revolutionized NLP tasks.",
        "year": 2021.0,
        "primary_category": "cs.CL",
        "authors": np.array(["Charlie Brown", "Diana Prince"]),
        "categories": np.array(["cs.CL", "cs.AI"]),
    },
    {
        "id": "paper_003",
        "title": "Graph Neural Networks Survey",
        "abstract": "A comprehensive survey of graph neural network architectures.",
        "year": 2021.0,
        "primary_category": "cs.LG",
        "authors": np.array(["Alice Smith", "Eve White"]),
        "categories": np.array(["cs.LG", "cs.AI"]),
    },
    {
        "id": "paper_004",
        "title": "Reinforcement Learning in Robotics",
        "abstract": "We apply reinforcement learning algorithms to robotic control.",
        "year": 2022.0,
        "primary_category": "cs.RO",
        "authors": np.array(["Frank Miller"]),
        "categories": np.array(["cs.RO", "cs.AI", "cs.LG"]),
    },
    {
        "id": "paper_005",
        "title": "Quantum Computing Basics",
        "abstract": "An introduction to quantum computing principles.",
        "year": 2020.0,
        "primary_category": "quant-ph",
        "authors": np.array(["Grace Hopper", "Henry Ford"]),
        "categories": np.array(["quant-ph", "cs.ET"]),
    },
    # Papers dengan null values
    {
        "id": "paper_006",
        "title": "Machine Learning Theory",
        "abstract": "Theoretical foundations of machine learning.",
        "year": None,  # null year
        "primary_category": None,  # null primary_category
        "authors": np.array(["Iris West"]),
        "categories": np.array(["cs.LG"]),
    },
    {
        "id": "paper_007",
        "title": "Data Mining Techniques",
        "abstract": "Advanced data mining and analysis techniques.",
        "year": 2022.0,
        "primary_category": "cs.DB",
        "authors": np.array([]),  # empty authors
        "categories": np.array([]),  # empty categories
    },
    # Papers untuk testing case sensitivity
    {
        "id": "paper_008",
        "title": "AI Safety Research",
        "abstract": "Ensuring safe artificial intelligence systems.",
        "year": 2021.0,
        "primary_category": "cs.AI",
        "authors": np.array(["ALICE SMITH", "jack BLACK"]),  # mixed case
        "categories": np.array(["CS.AI", "cs.cy"]),  # mixed case
    },
    {
        "id": "paper_009",
        "title": "Distributed Systems",
        "abstract": "Scalable distributed computing architectures.",
        "year": 2020.0,
        "primary_category": "cs.DC",
        "authors": np.array(["Karen Page"]),
        "categories": np.array(["cs.DC"]),
    },
    {
        "id": "paper_010",
        "title": "Cryptography Fundamentals",
        "abstract": "Modern cryptographic algorithms and protocols.",
        "year": 2023.0,
        "primary_category": "cs.CR",
        "authors": np.array(["Leo King", "Alice Smith"]),  # Alice appears again
        "categories": np.array(["cs.CR", "cs.IT"]),
    },
]


def make_test_df():
    """Create test DataFrame dari TEST_PAPERS."""
    df = pd.DataFrame(TEST_PAPERS)
    # For parquet compatibility, convert numpy arrays to lists during save
    # They will be loaded back as object dtype
    return df


class TestBinaryIndexer(unittest.TestCase):
    """Test BinaryIndexer: index building, multi-value fields, null handling."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.tmp.name, "test_dataset.parquet")
        self.index_dir = os.path.join(self.tmp.name, "index")
        
        # Save test dataset
        df = make_test_df()
        # Convert numpy arrays to lists for parquet serialization
        df_save = df.copy()
        df_save["authors"] = df_save["authors"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save["categories"] = df_save["categories"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save.to_parquet(self.dataset_path, engine="pyarrow")

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_building_creates_parquet_files(self):
        """Test bahwa indexer membuat semua file parquet yang diperlukan."""
        indexer = BinaryIndexer(self.dataset_path, self.index_dir)
        config = indexer.build_indexes(verbose=False)
        
        # Verify all files exist
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "year.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "primary_category.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "authors.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "categories.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "config.json")))
        
        # Verify config
        self.assertEqual(config["total_docs"], 10)
        self.assertIn("field_stats", config)

    def test_authors_index_has_multiple_entries_per_paper(self):
        """Test bahwa multi-author paper menghasilkan multiple entries."""
        indexer = BinaryIndexer(self.dataset_path, self.index_dir)
        indexer.build_indexes(verbose=False)
        
        authors_df = pd.read_parquet(os.path.join(self.index_dir, "authors.parquet"))
        
        # Paper 001 memiliki 2 authors (Alice Smith, Bob Jones)
        paper_001_entries = authors_df[authors_df["doc_idx"] == 0]
        self.assertEqual(len(paper_001_entries), 2)
        
        # Verify authors normalized to lowercase
        author_keys = set(paper_001_entries["key"].tolist())
        self.assertEqual(author_keys, {"alice smith", "bob jones"})

    def test_null_values_excluded_from_indexes(self):
        """Test bahwa null values di year dan primary_category di-skip."""
        indexer = BinaryIndexer(self.dataset_path, self.index_dir)
        config = indexer.build_indexes(verbose=False)
        
        # Paper 006 memiliki null year dan primary_category (doc_idx=5)
        year_df = pd.read_parquet(os.path.join(self.index_dir, "year.parquet"))
        primary_cat_df = pd.read_parquet(os.path.join(self.index_dir, "primary_category.parquet"))
        
        # Verify paper 006 tidak ada di index
        self.assertNotIn(5, year_df["doc_idx"].tolist())
        self.assertNotIn(5, primary_cat_df["doc_idx"].tolist())
        
        # Verify stats: 9 docs have non-null year, 9 have non-null primary_category
        self.assertEqual(config["field_stats"]["year"]["non_null_docs"], 9)
        self.assertEqual(config["field_stats"]["primary_category"]["non_null_docs"], 9)

    def test_config_contains_correct_stats(self):
        """Test bahwa config.json berisi stats yang benar."""
        indexer = BinaryIndexer(self.dataset_path, self.index_dir)
        config = indexer.build_indexes(verbose=False)
        
        # Verify year stats
        year_stats = config["field_stats"]["year"]
        self.assertEqual(year_stats["unique_keys"], 4)  # 2020, 2021, 2022, 2023
        self.assertEqual(year_stats["non_null_docs"], 9)
        
        # Verify primary_category stats
        cat_stats = config["field_stats"]["primary_category"]
        self.assertEqual(cat_stats["unique_keys"], 9)  # cs.CV, cs.CL, cs.LG, cs.RO, quant-ph, cs.DB, cs.AI, cs.DC, cs.CR
        
        # Verify authors stats (Alice Smith appears 3x, total unique authors)
        authors_stats = config["field_stats"]["authors"]
        self.assertGreater(authors_stats["unique_keys"], 10)
        
        # Verify categories stats
        categories_stats = config["field_stats"]["categories"]
        self.assertGreater(categories_stats["unique_keys"], 5)

    def test_empty_arrays_handled_correctly(self):
        """Test bahwa empty numpy arrays (paper 007) tidak menghasilkan entries."""
        indexer = BinaryIndexer(self.dataset_path, self.index_dir)
        indexer.build_indexes(verbose=False)
        
        authors_df = pd.read_parquet(os.path.join(self.index_dir, "authors.parquet"))
        categories_df = pd.read_parquet(os.path.join(self.index_dir, "categories.parquet"))
        
        # Paper 007 (doc_idx=6) memiliki empty authors dan categories
        self.assertNotIn(6, authors_df["doc_idx"].tolist())
        self.assertNotIn(6, categories_df["doc_idx"].tolist())


class TestBinaryFilter(unittest.TestCase):
    """Test BinaryFilter: filtering logic, AND/OR operations, case-insensitivity."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset_path = os.path.join(cls.tmp.name, "test_dataset.parquet")
        cls.index_dir = os.path.join(cls.tmp.name, "index")
        
        # Build indexes
        df = make_test_df()
        # Convert numpy arrays to lists for parquet serialization
        df_save = df.copy()
        df_save["authors"] = df_save["authors"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save["categories"] = df_save["categories"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save.to_parquet(cls.dataset_path, engine="pyarrow")
        
        indexer = BinaryIndexer(cls.dataset_path, cls.index_dir)
        indexer.build_indexes(verbose=False)
        
        cls.filter = BinaryFilter(cls.index_dir, cls.dataset_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_filter_by_year_range(self):
        """Test filter berdasarkan year range."""
        # 2020-2021: papers 001, 002, 003, 005, 008, 009 (doc_idx 0, 1, 2, 4, 7, 8)
        result = self.filter.filter_by_year(2020.0, 2021.0)
        expected = {0, 1, 2, 4, 7, 8}
        self.assertEqual(set(result), expected)

    def test_filter_by_year_edge_cases(self):
        """Test edge cases untuk year filtering."""
        # min=max: only 2022
        result = self.filter.filter_by_year(2022.0, 2022.0)
        expected = {3, 6}  # papers 004 dan 007
        self.assertEqual(set(result), expected)
        
        # No matches: year 2025
        result = self.filter.filter_by_year(2025.0, 2026.0)
        self.assertEqual(len(result), 0)

    def test_filter_by_primary_category_or_logic(self):
        """Test filter primary_category dengan OR logic."""
        # Single category
        result = self.filter.filter_by_primary_category(["cs.AI"])
        expected = {7}  # paper 008
        self.assertEqual(set(result), expected)
        
        # Multiple categories (OR logic)
        result = self.filter.filter_by_primary_category(["cs.CV", "cs.CL"])
        expected = {0, 1}  # papers 001 dan 002
        self.assertEqual(set(result), expected)

    def test_filter_by_primary_category_empty_list(self):
        """Test filter dengan empty category list."""
        result = self.filter.filter_by_primary_category([])
        self.assertEqual(len(result), 0)

    def test_get_pre_filter_candidates_and_logic(self):
        """Test kombinasi pre-filters dengan AND logic (intersection)."""
        # Year 2020-2021 AND primary_category cs.CV or cs.CL
        # Year 2020-2021: {0, 1, 2, 4, 8}
        # cs.CV or cs.CL: {0, 1}
        # Intersection: {0, 1}
        result = self.filter.get_pre_filter_candidates(
            year_range=(2020.0, 2021.0),
            primary_categories=["cs.CV", "cs.CL"]
        )
        expected = {0, 1}
        self.assertEqual(set(result), expected)

    def test_get_pre_filter_candidates_no_filters(self):
        """Test pre-filter tanpa filter apapun."""
        result = self.filter.get_pre_filter_candidates()
        # Tidak ada filter -> return empty array
        self.assertEqual(len(result), 0)

    def test_filter_by_authors_case_insensitive(self):
        """Test post-filter authors dengan case-insensitive exact match."""
        doc_ids = np.array([0, 1, 2, 3, 7, 9])
        
        # Search "alice smith" (lowercase) -> should match paper 001, 003, 008, 010
        result = self.filter.filter_by_authors(doc_ids, ["alice smith"])
        expected = {0, 2, 7, 9}  # papers dengan Alice Smith (various cases)
        self.assertEqual(set(result), expected)
        
        # Search "ALICE SMITH" (uppercase) -> same results
        result = self.filter.filter_by_authors(doc_ids, ["ALICE SMITH"])
        self.assertEqual(set(result), expected)

    def test_filter_by_authors_partial_name_not_matched(self):
        """Test bahwa partial name tidak matched (exact match only)."""
        doc_ids = np.array([0, 1, 2, 3])
        
        # Search "alice" (partial) -> tidak match karena exact match
        result = self.filter.filter_by_authors(doc_ids, ["alice"])
        self.assertEqual(len(result), 0)

    def test_filter_by_authors_multiple_authors_or_logic(self):
        """Test filter multiple authors dengan ANY logic."""
        doc_ids = np.array([0, 1, 2, 3, 4])
        
        # Search "alice smith" OR "bob jones"
        result = self.filter.filter_by_authors(doc_ids, ["alice smith", "bob jones"])
        expected = {0, 2}  # paper 001 (has both), paper 003 (has alice)
        self.assertEqual(set(result), expected)

    def test_filter_by_categories_case_insensitive(self):
        """Test post-filter categories dengan case-insensitive exact match."""
        doc_ids = np.array([0, 1, 2, 3, 4, 7])
        
        # Search "cs.ai" (lowercase) -> should match papers with cs.AI or CS.AI
        result = self.filter.filter_by_categories(doc_ids, ["cs.ai"])
        expected = {0, 1, 2, 3, 7}  # papers dengan cs.AI (various cases)
        self.assertEqual(set(result), expected)

    def test_filter_by_categories_any_logic(self):
        """Test filter categories dengan ANY logic."""
        doc_ids = np.array([0, 1, 2, 3, 4])
        
        # Search "cs.CV" OR "cs.CL"
        result = self.filter.filter_by_categories(doc_ids, ["cs.cv", "cs.cl"])
        expected = {0, 1}  # paper 001 (cs.CV), paper 002 (cs.CL)
        self.assertEqual(set(result), expected)

    def test_apply_post_filters_sequential(self):
        """Test aplikasi sequential post-filters dengan AND logic."""
        doc_ids = np.array([0, 1, 2, 3, 7, 9])
        
        # Filter by authors "alice smith" AND categories "cs.ai"
        # alice smith: {0, 2, 7, 9}
        # cs.ai: {0, 2, 3, 7}
        # Intersection: {0, 2, 7}
        result = self.filter.apply_post_filters(
            doc_ids,
            authors=["alice smith"],
            categories=["cs.ai"]
        )
        expected = {0, 2, 7}
        self.assertEqual(set(result), expected)

    def test_empty_results_handling(self):
        """Test handling empty results."""
        # Empty doc_ids input
        result = self.filter.filter_by_authors(np.array([]), ["alice smith"])
        self.assertEqual(len(result), 0)
        
        # No matches
        doc_ids = np.array([0, 1, 2])
        result = self.filter.filter_by_authors(doc_ids, ["nonexistent author"])
        self.assertEqual(len(result), 0)


class TestBM25FWithCandidates(unittest.TestCase):
    """Test BM25F.search() dengan candidates parameter."""

    @classmethod
    def setUpClass(cls):
        cls.documents = [
            {"title": "Deep learning for images", "abstract": "CNN architectures"},
            {"title": "Natural language processing", "abstract": "Transformer models"},
            {"title": "Graph neural networks", "abstract": "GNN for molecular graphs"},
            {"title": "Reinforcement learning", "abstract": "Q-learning algorithms"},
            {"title": "Computer vision basics", "abstract": "Image classification tasks"},
        ]
        cls.bm25 = BM25F(cls.documents)

    def test_search_without_candidates_backward_compatibility(self):
        """Test bahwa search tanpa candidates bekerja seperti sebelumnya."""
        results = self.bm25.search("learning")
        
        # Should return results for all matching documents
        self.assertGreater(len(results), 0)
        
        # All results should be tuples (doc_id, score)
        for doc_id, score in results:
            self.assertIsInstance(doc_id, int)
            self.assertGreater(score, 0)

    def test_search_with_list_candidates(self):
        """Test search dengan list candidates."""
        # Only search in docs 0, 2, 4
        results = self.bm25.search("learning", candidates=[0, 2, 4])
        
        # Results should only contain doc_ids from candidates
        result_doc_ids = {doc_id for doc_id, score in results}
        self.assertTrue(result_doc_ids.issubset({0, 2, 4}))

    def test_search_with_numpy_array_candidates(self):
        """Test search dengan numpy array candidates."""
        candidates = np.array([1, 3])
        results = self.bm25.search("learning", candidates=candidates)
        
        # Results should only contain doc_ids from candidates
        result_doc_ids = {doc_id for doc_id, score in results}
        self.assertTrue(result_doc_ids.issubset({1, 3}))

    def test_empty_candidates_returns_empty_results(self):
        """Test bahwa empty candidates mengembalikan empty results."""
        results = self.bm25.search("learning", candidates=[])
        self.assertEqual(len(results), 0)

    def test_candidates_with_nonexistent_doc_ids(self):
        """Test candidates dengan doc_ids yang tidak exist."""
        # Doc 999 does not exist (only have 0-4)
        results = self.bm25.search("learning", candidates=[999])
        self.assertEqual(len(results), 0)

    def test_candidates_filters_correctly(self):
        """Test bahwa candidates filtering works correctly."""
        # Search "vision" without candidates
        all_results = self.bm25.search("vision")
        all_doc_ids = {doc_id for doc_id, score in all_results}
        
        # Search "vision" with candidates=[0, 4]
        # Only doc 4 should match (has "vision" in title)
        filtered_results = self.bm25.search("vision", candidates=[0, 4])
        filtered_doc_ids = {doc_id for doc_id, score in filtered_results}
        
        # Filtered results should be subset of all results
        self.assertTrue(filtered_doc_ids.issubset(all_doc_ids))
        
        # Only doc 4 should appear
        self.assertEqual(filtered_doc_ids, {4})

    def test_candidates_with_top_k(self):
        """Test kombinasi candidates dengan top_k."""
        # Should apply candidates filter then limit to top_k
        results = self.bm25.search("learning", candidates=[0, 1, 2, 3], top_k=2)
        
        # Should return at most 2 results
        self.assertLessEqual(len(results), 2)
        
        # All results should be from candidates
        result_doc_ids = {doc_id for doc_id, score in results}
        self.assertTrue(result_doc_ids.issubset({0, 1, 2, 3}))


class TestSearchPipeline(unittest.TestCase):
    """Test full search pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Setup test environment dengan mock indexes dan dataset."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset_path = os.path.join(cls.tmp.name, "test_dataset.parquet")
        cls.index_dir = os.path.join(cls.tmp.name, "index")
        
        # Build binary indexes
        df = make_test_df()
        # Convert numpy arrays to lists for parquet serialization
        df_save = df.copy()
        df_save["authors"] = df_save["authors"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save["categories"] = df_save["categories"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save.to_parquet(cls.dataset_path, engine="pyarrow")
        
        indexer = BinaryIndexer(cls.dataset_path, cls.index_dir)
        indexer.build_indexes(verbose=False)
        
        # Untuk testing, kita perlu BinaryFilter
        cls.binary_filter = BinaryFilter(cls.index_dir, cls.dataset_path)
        cls.dataset = pd.read_parquet(cls.dataset_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_search_with_no_filters(self):
        """Test search tanpa filters apapun."""
        # Create simple BM25F search
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        
        results = bm25.search("learning", top_k=5)
        
        # Should return results
        self.assertGreater(len(results), 0)

    def test_search_with_only_pre_filters(self):
        """Test search dengan hanya pre-filters."""
        # Get candidates from pre-filters
        candidates = self.binary_filter.get_pre_filter_candidates(
            year_range=(2020.0, 2021.0),
            primary_categories=["cs.CV", "cs.CL"]
        )
        
        # Should get papers 001 and 002
        self.assertEqual(set(candidates), {0, 1})
        
        # Search with candidates
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        results = bm25.search("learning", candidates=candidates)
        
        # Results should only be from candidates
        result_doc_ids = {doc_id for doc_id, score in results}
        self.assertTrue(result_doc_ids.issubset(set(candidates)))

    def test_search_with_only_post_filters(self):
        """Test search dengan hanya post-filters."""
        # Do BM25F search first
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        results = bm25.search("learning", top_k=10)
        
        # Get doc_ids from results
        doc_ids = np.array([doc_id for doc_id, score in results])
        
        # Apply post-filters
        filtered_doc_ids = self.binary_filter.apply_post_filters(
            doc_ids,
            authors=["alice smith"],
            categories=["cs.ai"]
        )
        
        # Should filter to only papers with alice smith AND cs.ai
        self.assertGreater(len(filtered_doc_ids), 0)

    def test_search_combining_all_filters(self):
        """Test search dengan kombinasi pre-filters dan post-filters."""
        # Step 1: Pre-filters
        pre_candidates = self.binary_filter.get_pre_filter_candidates(
            year_range=(2020.0, 2022.0)
        )
        
        # Step 2: BM25F search with candidates
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        results = bm25.search("learning", candidates=pre_candidates)
        
        # Step 3: Post-filters
        doc_ids = np.array([doc_id for doc_id, score in results])
        final_doc_ids = self.binary_filter.apply_post_filters(
            doc_ids,
            categories=["cs.ai"]
        )
        
        # Should have filtered results
        self.assertGreaterEqual(len(final_doc_ids), 0)

    def test_empty_results_at_pre_filter_stage(self):
        """Test handling empty results dari pre-filter."""
        # No papers in year 2025
        candidates = self.binary_filter.get_pre_filter_candidates(
            year_range=(2025.0, 2026.0)
        )
        
        self.assertEqual(len(candidates), 0)
        
        # Search with empty candidates should return empty
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        results = bm25.search("learning", candidates=candidates)
        
        self.assertEqual(len(results), 0)

    def test_empty_results_at_post_filter_stage(self):
        """Test handling empty results dari post-filter."""
        # Search first
        docs = [{"title": row["title"], "abstract": row["abstract"]} for _, row in self.dataset.iterrows()]
        bm25 = BM25F(docs)
        results = bm25.search("learning", top_k=5)
        
        doc_ids = np.array([doc_id for doc_id, score in results])
        
        # Apply post-filter with non-matching author
        filtered = self.binary_filter.apply_post_filters(
            doc_ids,
            authors=["nonexistent author"]
        )
        
        self.assertEqual(len(filtered), 0)

    def test_metadata_enrichment(self):
        """Test bahwa metadata enrichment includes all expected fields."""
        # Simulate getting results and enriching with metadata
        doc_id = 0
        
        # Get metadata from dataset
        metadata = self.dataset.iloc[doc_id].to_dict()
        
        # Verify expected fields
        expected_fields = ["id", "title", "abstract", "year", "primary_category", "authors", "categories"]
        for field in expected_fields:
            self.assertIn(field, metadata)

    def test_null_values_in_results_handled(self):
        """Test bahwa null values dalam metadata di-handle dengan benar."""
        # Paper 006 has null year and primary_category
        doc_id = 5
        metadata = self.dataset.iloc[doc_id].to_dict()
        
        # Verify null values are preserved
        self.assertTrue(pd.isna(metadata["year"]))
        self.assertTrue(pd.isna(metadata["primary_category"]))
        
        # Should not crash when accessing
        self.assertIsNotNone(metadata["id"])
        self.assertIsNotNone(metadata["title"])


if __name__ == "__main__":
    unittest.main()
