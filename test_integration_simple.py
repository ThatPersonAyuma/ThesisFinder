"""
Simple integration test for Binary Model components.
"""
import json
from pathlib import Path

import numpy as np

from src.algorithm.bm25f import BM25F
from src.binary.config import BINARY_INDEX_DIR
from src.binary.filter import BinaryFilter
from src.vsm.tfidf import INDEX_DIR as VSM_INDEX_DIR


def test_binary_filter_standalone():
    """Test binary filter with actual indexes."""
    print("=" * 80)
    print("INTEGRATION TEST: Binary Filter Standalone")
    print("=" * 80)

    print("\n1. Initializing Binary Filter...")
    bf = BinaryFilter()
    print(f"   {bf}")

    print("\n2. Testing pre-filters...")
    year_2020_2021 = bf.get_pre_filter_candidates(year_range=(2020.0, 2021.0))
    print(f"   Year 2020-2021: {len(year_2020_2021):,} candidates")

    cs_ai = bf.get_pre_filter_candidates(primary_categories=["cs.AI"])
    print(f"   Primary category cs.AI: {len(cs_ai):,} candidates")

    combined = bf.get_pre_filter_candidates(
        year_range=(2020.0, 2021.0),
        primary_categories=["cs.AI", "cs.LG"],
    )
    print(f"   Year 2020-2021 AND (cs.AI OR cs.LG): {len(combined):,} candidates")

    print("\n3. Testing post-filters...")
    test_docs = year_2020_2021[:1000]
    print(f"   Test set: {len(test_docs)} docs")

    cs_lg_filtered = bf.filter_by_categories(test_docs, ["cs.lg"])
    print(f"   After category filter (cs.lg): {len(cs_lg_filtered)} docs")

    filtered_authors = bf.filter_by_authors(test_docs, ["nonexistent author xyz"])
    print(f"   After author filter (nonexistent): {len(filtered_authors)} docs (expected: 0)")

    final = bf.apply_post_filters(test_docs, categories=["cs.ai"])
    print(f"   Combined post-filters (cs.ai): {len(final)} docs")

    print("\n4. Testing edge cases...")
    empty = bf.get_pre_filter_candidates(year_range=(2050.0, 2051.0))
    print(f"   Future year range: {len(empty)} candidates (expected: 0)")
    assert len(empty) == 0

    empty_post = bf.apply_post_filters(np.array([], dtype=np.int64), authors=["test"])
    print(f"   Post-filter empty input: {len(empty_post)} docs (expected: 0)")
    assert len(empty_post) == 0


def test_bm25f_with_candidates():
    """Test BM25F with candidates parameter using small dataset."""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: BM25F with Candidates")
    print("=" * 80)

    test_docs = [
        {"title": "Deep Learning for Computer Vision", "abstract": "Neural networks for image classification"},
        {"title": "Natural Language Processing", "abstract": "Transformers and attention mechanisms"},
        {"title": "Graph Neural Networks", "abstract": "Message passing on molecular graphs"},
        {"title": "Reinforcement Learning", "abstract": "Q-learning and policy gradients"},
        {"title": "Computer Vision Basics", "abstract": "Image processing and feature extraction"},
        {"title": "Machine Learning Theory", "abstract": "Statistical learning and generalization"},
        {"title": "Deep Neural Networks", "abstract": "Backpropagation and optimization"},
        {"title": "Attention Mechanisms", "abstract": "Self-attention and transformers"},
        {"title": "Graph Algorithms", "abstract": "Shortest paths and network analysis"},
        {"title": "Probabilistic Models", "abstract": "Bayesian inference and graphical models"},
    ]

    bm25 = BM25F(test_docs)
    results_filtered = bm25.search("deep learning neural networks", top_k=5, candidates=[0, 1, 2, 3])
    assert {doc_id for doc_id, _ in results_filtered}.issubset({0, 1, 2, 3})

    results_array = bm25.search("learning optimization", top_k=3, candidates=np.array([4, 5, 6]))
    assert {doc_id for doc_id, _ in results_array}.issubset({4, 5, 6})

    assert bm25.search("test", candidates=[]) == []
    print("   BM25F candidates verification PASSED")


def test_index_consistency():
    """Check if VSM and Binary indexes are built on the same dataset."""
    print("\n" + "=" * 80)
    print("INDEX CONSISTENCY CHECK")
    print("=" * 80)

    vsm_config_path = Path(VSM_INDEX_DIR) / "config.json"
    binary_config_path = Path(BINARY_INDEX_DIR) / "config.json"

    vsm_config = json.loads(vsm_config_path.read_text(encoding="utf-8")) if vsm_config_path.exists() else None
    binary_config = json.loads(binary_config_path.read_text(encoding="utf-8")) if binary_config_path.exists() else None

    print(f"VSM Index: {vsm_config['n_docs']:,} docs" if vsm_config else "VSM Index: Not found")
    print(f"Binary Index: {binary_config['total_docs']:,} docs" if binary_config else "Binary Index: Not found")

    if vsm_config and binary_config:
        assert vsm_config["n_docs"] == binary_config["total_docs"], (
            f"Index mismatch: VSM={vsm_config['n_docs']} != Binary={binary_config['total_docs']}"
        )
        print("Indexes are consistent (same number of documents)")


if __name__ == "__main__":
    test_binary_filter_standalone()
    test_bm25f_with_candidates()
    test_index_consistency()
    print("\nALL INTEGRATION TESTS PASSED")
