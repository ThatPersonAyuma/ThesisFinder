"""Menjalankan seluruh unit test BM25F plus pengujian fitur integrasi VSM-BM25F."""

import importlib.util
import unittest
from pathlib import Path

from src.algorithm.bm25f import BM25F
from src.vsm.tfidf import analyze_text

_BM25_TEST_PATH = Path(__file__).resolve().parents[1] / "src" / "algorithm" / "bm25-test.py"
_SPEC = importlib.util.spec_from_file_location("bm25_test_suite", _BM25_TEST_PATH)
_BM25_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BM25_MOD)

TestBM25Formulas = _BM25_MOD.TestBM25Formulas
TestBM25SearchProperties = _BM25_MOD.TestBM25SearchProperties
TestBM25FFields = _BM25_MOD.TestBM25FFields
TestBM25QueryDataset = _BM25_MOD.TestBM25QueryDataset


class TestBM25FVSMIntegration(unittest.TestCase):
    def test_custom_vsm_tokenizer_and_rerank(self):
        candidate_docs = [
            (10, {"title": "Deep Learning Models", "abstract": "Neural networks for images."}),
            (25, {"title": "Graph Neural Networks", "abstract": "Molecular graph modeling and studies."}),
            (42, {"title": "Graph Algorithms Survey", "abstract": "Shortest path routing."}),
        ]
        ranked = BM25F.rerank(
            "neural model studied",
            candidate_docs,
            tokenizer=analyze_text,
            corpus_idf={"neural": 3.0, "model": 1.5, "study": 2.0},
            top_k=2,
        )
        self.assertEqual(len(ranked), 2)
        doc_ids = [doc_id for doc_id, _ in ranked]
        self.assertTrue(set(doc_ids).issubset({10, 25, 42}))
        self.assertEqual(ranked[0][0], 25)
        self.assertGreater(ranked[0][1], ranked[1][1])


if __name__ == "__main__":
    unittest.main()
