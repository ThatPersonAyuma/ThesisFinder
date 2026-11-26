"""
Demo script untuk mencoba Binary Model API
"""
from src.binary.filter import BinaryFilter
from src.vsm.searcher import VSMSearcher
import numpy as np

print("="*60)
print("DEMO BINARY MODEL - ThesisFinder")
print("="*60)

# Initialize components
print("\n1. Inisialisasi Binary Filter...")
binary_filter = BinaryFilter()
print("   ✓ Binary Filter loaded")

print("\n2. Inisialisasi VSM Searcher...")
vsm_searcher = VSMSearcher()
print("   ✓ VSM Searcher loaded")

# Demo 1: Pre-filtering by year
print("\n" + "="*60)
print("DEMO 1: Pre-filtering by Year Range")
print("="*60)
candidates_2024 = binary_filter.filter_by_year(2024.0, 2024.0)
print(f"Papers published in 2024: {len(candidates_2024):,}")

candidates_2020_2022 = binary_filter.filter_by_year(2020.0, 2022.0)
print(f"Papers published 2020-2022: {len(candidates_2020_2022):,}")

# Demo 2: Pre-filtering by primary category
print("\n" + "="*60)
print("DEMO 2: Pre-filtering by Primary Category")
print("="*60)
candidates_ai = binary_filter.filter_by_primary_category(['cs.AI'])
print(f"Papers with primary_category=cs.AI: {len(candidates_ai):,}")

candidates_ai_lg = binary_filter.filter_by_primary_category(['cs.AI', 'cs.LG'])
print(f"Papers with cs.AI or cs.LG: {len(candidates_ai_lg):,}")

# Demo 3: Combined pre-filtering
print("\n" + "="*60)
print("DEMO 3: Combined Pre-filtering (Year AND Category)")
print("="*60)
candidates = binary_filter.get_pre_filter_candidates(
    year_range=(2023.0, 2024.0),
    primary_categories=['cs.AI', 'cs.LG']
)
print(f"Papers from 2023-2024 with cs.AI/cs.LG: {len(candidates):,}")

# Demo 4: Full pipeline with VSM search
print("\n" + "="*60)
print("DEMO 4: Full Pipeline - Pre-filter → VSM Search → Post-filter")
print("="*60)

query = "deep learning neural networks"
print(f"Query: '{query}'")

# Step 1: Pre-filter
print("\nStep 1: Pre-filtering (2023-2024, cs.AI/cs.LG)")
candidates = binary_filter.get_pre_filter_candidates(
    year_range=(2023.0, 2024.0),
    primary_categories=['cs.AI', 'cs.LG']
)
print(f"   Pre-filter candidates: {len(candidates):,}")

# Step 2: VSM search with candidates
print("\nStep 2: VSM search (top 10)")
results = vsm_searcher.search(query, top_k=10, candidates=candidates)
print(f"   VSM results: {len(results)}")

# Display results
print("\nTop 5 Results:")
for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. [{result.score:.3f}] {result.title[:80]}...")
    print(f"   ID: {result.id}")

# Demo 5: Post-filtering on results
print("\n" + "="*60)
print("DEMO 5: Post-filtering (Filter by Categories)")
print("="*60)

# Get doc_ids from VSM results
doc_ids = np.array([r.doc_idx for r in results])
print(f"VSM returned {len(doc_ids)} doc_ids")

# Post-filter by categories
filtered_doc_ids = binary_filter.filter_by_categories(
    doc_ids, 
    categories=['cs.CL', 'cs.CV']  # Only papers with these categories
)
print(f"After category filter (cs.CL or cs.CV): {len(filtered_doc_ids)} doc_ids")

# Show filtered results
if len(filtered_doc_ids) > 0:
    print("\nFiltered Results:")
    for i, doc_idx in enumerate(filtered_doc_ids[:3], 1):
        result = next(r for r in results if r.doc_idx == doc_idx)
        metadata = binary_filter.dataset.iloc[doc_idx]
        print(f"\n{i}. {result.title[:80]}...")
        print(f"   Categories: {metadata['categories']}")
        print(f"   Authors: {', '.join(metadata['authors'][:3])}")
        print(f"   Year: {metadata['year']}")
else:
    print("No results after filtering")

print("\n" + "="*60)
print("DEMO SELESAI!")
print("="*60)
