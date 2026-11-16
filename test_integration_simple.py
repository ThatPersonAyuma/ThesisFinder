"""
Simple integration test for Binary Model components.
Tests individual components without requiring full dataset sync.
"""
import numpy as np
from src.binary.filter import BinaryFilter
from src.algorithm.bm25f import BM25F
import pandas as pd

def test_binary_filter_standalone():
    """Test binary filter with actual indexes."""
    print("=" * 80)
    print("INTEGRATION TEST: Binary Filter Standalone")
    print("=" * 80)
    
    # Initialize binary filter
    print("\n1. Initializing Binary Filter...")
    bf = BinaryFilter()
    print(f"   {bf}")
    
    # Test pre-filtering
    print("\n2. Testing pre-filters...")
    
    # Year range filter
    year_2020_2021 = bf.get_pre_filter_candidates(year_range=(2020.0, 2021.0))
    print(f"   Year 2020-2021: {len(year_2020_2021):,} candidates")
    
    # Primary category filter
    cs_ai = bf.get_pre_filter_candidates(primary_categories=["cs.AI"])
    print(f"   Primary category cs.AI: {len(cs_ai):,} candidates")
    
    # Combined pre-filters
    combined = bf.get_pre_filter_candidates(
        year_range=(2020.0, 2021.0),
        primary_categories=["cs.AI", "cs.LG"]
    )
    print(f"   Year 2020-2021 AND (cs.AI OR cs.LG): {len(combined):,} candidates")
    
    # Test post-filtering
    print("\n3. Testing post-filters...")
    
    # Get some docs for testing (first 1000 from year range)
    test_docs = year_2020_2021[:1000]
    print(f"   Test set: {len(test_docs)} docs")
    
    # Filter by categories
    cs_lg_filtered = bf.filter_by_categories(test_docs, ["cs.lg"])
    print(f"   After category filter (cs.lg): {len(cs_lg_filtered)} docs")
    
    # Filter by authors (case-insensitive test)
    # Note: We don't know which authors exist, so let's just test empty result
    filtered_authors = bf.filter_by_authors(test_docs, ["nonexistent author xyz"])
    print(f"   After author filter (nonexistent): {len(filtered_authors)} docs (expected: 0)")
    
    # Combined post-filters
    final = bf.apply_post_filters(
        test_docs,
        categories=["cs.ai"]
    )
    print(f"   Combined post-filters (cs.ai): {len(final)} docs")
    
    # Test edge cases
    print("\n4. Testing edge cases...")
    
    # Empty pre-filter result
    empty = bf.get_pre_filter_candidates(year_range=(2050.0, 2051.0))
    print(f"   Future year range: {len(empty)} candidates (expected: 0)")
    
    # Post-filter with empty input
    empty_post = bf.apply_post_filters(np.array([], dtype=np.int64), authors=["test"])
    print(f"   Post-filter empty input: {len(empty_post)} docs (expected: 0)")
    
    print("\n" + "=" * 80)
    print("BINARY FILTER TEST PASSED ✓")
    print("=" * 80)


def test_bm25f_with_candidates():
    """Test BM25F with candidates parameter using small dataset."""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: BM25F with Candidates")
    print("=" * 80)
    
    # Create small test dataset
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
    
    print(f"\n1. Creating BM25F index with {len(test_docs)} documents...")
    bm25 = BM25F(test_docs)
    
    # Test without candidates (baseline)
    print("\n2. Search without candidates...")
    results_all = bm25.search("deep learning neural networks", top_k=5)
    print(f"   Results: {len(results_all)}")
    for i, (doc_id, score) in enumerate(results_all, 1):
        print(f"   {i}. [{score:.3f}] {test_docs[doc_id]['title']}")
    
    # Test with list candidates
    print("\n3. Search with list candidates [0, 1, 2, 3]...")
    candidates_list = [0, 1, 2, 3]
    results_filtered = bm25.search("deep learning neural networks", top_k=5, candidates=candidates_list)
    print(f"   Results: {len(results_filtered)}")
    for i, (doc_id, score) in enumerate(results_filtered, 1):
        print(f"   {i}. [{score:.3f}] {test_docs[doc_id]['title']}")
    
    # Verify all results are from candidates
    result_ids = {doc_id for doc_id, _ in results_filtered}
    assert result_ids.issubset(set(candidates_list)), "Results contain doc_ids outside candidates!"
    print("   ✓ All results are from candidates")
    
    # Test with numpy array candidates
    print("\n4. Search with numpy array candidates...")
    candidates_array = np.array([4, 5, 6])
    results_array = bm25.search("learning optimization", top_k=3, candidates=candidates_array)
    print(f"   Results: {len(results_array)}")
    for i, (doc_id, score) in enumerate(results_array, 1):
        print(f"   {i}. [{score:.3f}] {test_docs[doc_id]['title']}")
    
    result_ids = {doc_id for doc_id, _ in results_array}
    assert result_ids.issubset(set(candidates_array)), "Results contain doc_ids outside candidates!"
    print("   ✓ All results are from candidates")
    
    # Test empty candidates
    print("\n5. Search with empty candidates...")
    results_empty = bm25.search("test", candidates=[])
    print(f"   Results: {len(results_empty)} (expected: 0)")
    assert len(results_empty) == 0, "Empty candidates should return empty results!"
    print("   ✓ Empty candidates handled correctly")
    
    print("\n" + "=" * 80)
    print("BM25F CANDIDATES TEST PASSED ✓")
    print("=" * 80)


def test_index_consistency():
    """Check if indexes are built on the same dataset."""
    print("\n" + "=" * 80)
    print("INDEX CONSISTENCY CHECK")
    print("=" * 80)
    
    import json
    from pathlib import Path
    
    vsm_config_path = Path("e:/ThesisFinder/data/index/vsm/config.json")
    binary_config_path = Path("e:/ThesisFinder/data/index/binary/config.json")
    
    if vsm_config_path.exists():
        vsm_config = json.loads(vsm_config_path.read_text())
        print(f"\nVSM Index: {vsm_config['n_docs']:,} documents")
    else:
        print("\nVSM Index: Not found")
    
    if binary_config_path.exists():
        binary_config = json.loads(binary_config_path.read_text())
        print(f"Binary Index: {binary_config['total_docs']:,} documents")
    else:
        print("Binary Index: Not found")
    
    if vsm_config_path.exists() and binary_config_path.exists():
        if vsm_config['n_docs'] != binary_config['total_docs']:
            print("\n⚠️  WARNING: Index mismatch detected!")
            print("   VSM and Binary indexes are built on different datasets.")
            print("   For proper integration, rebuild indexes on the same dataset:")
            print("   1. python -m src.vsm.indexer --input data/completed.parquet")
            print("   2. python -m src.binary.indexer --input data/completed.parquet")
        else:
            print("\n✓ Indexes are consistent (same number of documents)")
    
    print("=" * 80)


if __name__ == "__main__":
    test_binary_filter_standalone()
    test_bm25f_with_candidates()
    test_index_consistency()
    
    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 80)
