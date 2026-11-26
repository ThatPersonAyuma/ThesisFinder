"""Integration tests for SearchPipeline (Binary Pre-Filter -> VSM -> BM25F -> Binary Post-Filter)."""

import os
import tempfile
import unittest

import pandas as pd

from src.binary.indexer import BinaryIndexer
from src.pipeline.search_pipeline import SearchPipeline
from src.vsm.indexer import build_index_from_parquet

PAPERS = [
    {
        "id": "http://arxiv.org/abs/2001.00001v1",
        "title": "Graph Neural Networks for Molecules",
        "abstract": "We study graph neural networks for molecular property prediction.",
        "year": 2020.0,
        "primary_category": "cs.LG",
        "authors": ["Alice Smith", "Bob Jones"],
        "categories": ["cs.LG", "cs.AI"],
        "keywords": "graph neural networks, molecules",
    },
    {
        "id": "http://arxiv.org/abs/2101.00002v1",
        "title": "Deep Graph Representation Learning",
        "abstract": "Graph representation learning improves graph classification tasks.",
        "year": 2021.0,
        "primary_category": "cs.LG",
        "authors": ["Alice Smith"],
        "categories": ["cs.LG"],
        "keywords": "graph learning",
    },
    {
        "id": "http://arxiv.org/abs/2102.00003v1",
        "title": "Graph Algorithms in Practice",
        "abstract": "Shortest path graph algorithms and routing optimization.",
        "year": 2021.0,
        "primary_category": "cs.DS",
        "authors": ["Charlie Brown"],
        "categories": ["cs.DS", "cs.LG"],
        "keywords": "shortest path",
    },
    {
        "id": "http://arxiv.org/abs/2201.00004v1",
        "title": "Graph Convolutions for Vision",
        "abstract": "Applying graph neural concepts to computer vision.",
        "year": 2022.0,
        "primary_category": "cs.CV",
        "authors": ["Bob Jones"],
        "categories": ["cs.CV", "cs.LG"],
        "keywords": "vision, graph",
    },
    {
        "id": "http://arxiv.org/abs/2301.00005v1",
        "title": "Graph Models by Alice",
        "abstract": "We propose graph models for protein folding.",
        "year": 2021.0,
        "primary_category": "cs.LG",
        "authors": ["Alice Smith", "Dan Lee"],
        "categories": ["cs.LG", "q-bio.BM"],
        "keywords": "protein",
    },
]


class TestIntegratedSearchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset_path = os.path.join(cls.tmp.name, "papers.parquet")
        cls.structure_path = os.path.join(cls.tmp.name, "structure.parquet")
        cls.vsm_dir = os.path.join(cls.tmp.name, "vsm_idx")
        cls.binary_dir = os.path.join(cls.tmp.name, "bin_idx")

        pd.DataFrame(PAPERS).to_parquet(cls.dataset_path, engine="pyarrow", index=False)
        pd.DataFrame([
            {
                "id": "http://arxiv.org/abs/2001.00001v1",
                "keywords": "molecular graph benchmark",
                "metode": "message passing neural architecture",
                "evaluasi": "state of the art drug accuracy",
                "kesimpulan": "strong molecular generalization",
            }
        ]).to_parquet(cls.structure_path, engine="pyarrow", index=False)

        build_index_from_parquet(cls.dataset_path, cls.vsm_dir, verbose=False)
        BinaryIndexer(cls.dataset_path, cls.binary_dir).build_indexes(verbose=False)

        cls.pipeline = SearchPipeline(
            dataset_path=cls.dataset_path,
            vsm_index_dir=cls.vsm_dir,
            binary_index_dir=cls.binary_dir,
            structure_path=cls.structure_path,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hybrid_vsm_and_bm25f_modes(self):
        for mode in ("hybrid", "vsm", "bm25f"):
            results = self.pipeline.search("graph neural molecules", mode=mode, top_k=3)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["doc_idx"], 0)
            self.assertGreater(results[0]["score"], 0)
            if mode == "hybrid":
                self.assertIsNotNone(results[0]["bm25f_score"])
                self.assertGreater(results[0]["vsm_score"], 0)

    def test_pre_and_post_filtering_no_top_k_starvation(self):
        # Hanya meminta top_k=2 namun ada dokumen peringkat bawah yang punya author "Bob Jones".
        results = self.pipeline.search(
            "graph",
            year_range=(2020.0, 2022.0),
            authors=["bob jones"],
            top_k=2,
            mode="hybrid",
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("Bob Jones", r["authors"])
            self.assertTrue(2020 <= r["year"] <= 2022)

    def test_structure_fields_used_in_bm25f_rerank(self):
        # "message passing" hanya ada di file structure.parquet milik doc 0, tetap cocok di title/abstract via "graph"
        results = self.pipeline.search("graph message passing", mode="hybrid", top_k=3)
        self.assertEqual(results[0]["doc_idx"], 0)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(self.pipeline.search("", top_k=5), [])
        self.assertEqual(self.pipeline.search("graph", year_range=(2099.0, 2100.0)), [])
        self.assertEqual(self.pipeline.search("graph", authors=["Unknown Person"]), [])
        with self.assertRaises(ValueError):
            self.pipeline.search("graph", mode="invalid_mode")


if __name__ == "__main__":
    unittest.main()
