"""
Demo script untuk Binary Model (tanpa VSM integration)
Showcases filtering functionality saja
"""
from src.binary.filter import BinaryFilter
import numpy as np

print("="*70)
print("DEMO BINARY MODEL - ThesisFinder (Standalone Filtering)")
print("="*70)

# Initialize Binary Filter
print("\n✓ Inisialisasi Binary Filter...")
bf = BinaryFilter()
print(f"  Dataset: {len(bf.dataset):,} papers")
print(f"  Year index: {len(bf.year_index):,} entries")
print(f"  Primary category index: {len(bf.primary_category_index):,} entries")
print(f"  Authors index: {len(bf.authors_index):,} entries")
print(f"  Categories index: {len(bf.categories_index):,} entries")

# Demo 1: Year filtering
print("\n" + "="*70)
print("DEMO 1: Year Range Filtering")
print("="*70)

year_2024 = bf.filter_by_year(2024.0, 2024.0)
print(f"✓ Papers in 2024: {len(year_2024):,}")

year_2020_2022 = bf.filter_by_year(2020.0, 2022.0)
print(f"✓ Papers in 2020-2022: {len(year_2020_2022):,}")

year_2015_2017 = bf.filter_by_year(2015.0, 2017.0)
print(f"✓ Papers in 2015-2017: {len(year_2015_2017):,}")

# Show sample papers from 2024
print(f"\nSample papers from 2024 (first 3):")
for i, doc_idx in enumerate(year_2024[:3], 1):
    paper = bf.dataset.iloc[doc_idx]
    print(f"\n{i}. {paper['title'][:70]}...")
    print(f"   Year: {paper['year']}, ID: {paper['id']}")

# Demo 2: Primary Category filtering
print("\n" + "="*70)
print("DEMO 2: Primary Category Filtering (OR logic)")
print("="*70)

cat_ai = bf.filter_by_primary_category(['cs.AI'])
print(f"✓ Papers with primary_category = cs.AI: {len(cat_ai):,}")

cat_lg = bf.filter_by_primary_category(['cs.LG'])
print(f"✓ Papers with primary_category = cs.LG: {len(cat_lg):,}")

cat_ai_lg = bf.filter_by_primary_category(['cs.AI', 'cs.LG'])
print(f"✓ Papers with cs.AI OR cs.LG: {len(cat_ai_lg):,}")
print(f"  (Union of cs.AI and cs.LG)")

# Demo 3: Combined Pre-filtering (AND logic)
print("\n" + "="*70)
print("DEMO 3: Combined Pre-filtering (Year AND Primary Category)")
print("="*70)

# 2024 papers in cs.AI
candidates_1 = bf.get_pre_filter_candidates(
    year_range=(2024.0, 2024.0),
    primary_categories=['cs.AI']
)
print(f"✓ Papers: Year=2024 AND Category=cs.AI")
print(f"  Result: {len(candidates_1):,} papers")

# 2020-2022 papers in cs.AI or cs.LG
candidates_2 = bf.get_pre_filter_candidates(
    year_range=(2020.0, 2022.0),
    primary_categories=['cs.AI', 'cs.LG']
)
print(f"\n✓ Papers: Year=2020-2022 AND Category=(cs.AI OR cs.LG)")
print(f"  Result: {len(candidates_2):,} papers")

# Show sample
print(f"\nSample papers from 2020-2022 with cs.AI/cs.LG (first 3):")
for i, doc_idx in enumerate(candidates_2[:3], 1):
    paper = bf.dataset.iloc[doc_idx]
    print(f"\n{i}. {paper['title'][:70]}...")
    print(f"   Year: {paper['year']}, Category: {paper['primary_category']}")
    print(f"   Authors: {', '.join(paper['authors'][:2])}")

# Demo 4: Post-filtering by Authors
print("\n" + "="*70)
print("DEMO 4: Post-filtering by Authors (Case-insensitive)")
print("="*70)

# Ambil sample doc_ids untuk di-filter
sample_docs = year_2020_2022[:100]  # 100 papers from 2020-2022
print(f"Starting with {len(sample_docs)} papers from 2020-2022")

# Filter by specific authors
filtered_bengio = bf.filter_by_authors(sample_docs, ['yoshua bengio'])
print(f"\n✓ Papers by 'Yoshua Bengio': {len(filtered_bengio)}")

if len(filtered_bengio) > 0:
    for i, doc_idx in enumerate(filtered_bengio[:3], 1):
        paper = bf.dataset.iloc[doc_idx]
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Year: {paper['year']}")

# Demo 5: Post-filtering by Categories
print("\n" + "="*70)
print("DEMO 5: Post-filtering by Categories (ANY logic)")
print("="*70)

# Get cs.AI papers from 2024
ai_2024 = bf.get_pre_filter_candidates(
    year_range=(2024.0, 2024.0),
    primary_categories=['cs.AI']
)
print(f"Starting with {len(ai_2024):,} cs.AI papers from 2024")

# Post-filter: only papers that also have cs.CV or cs.CL category
sample_ai = ai_2024[:500]  # Take first 500 for demo
filtered_cv_cl = bf.filter_by_categories(sample_ai, ['cs.CV', 'cs.CL'])
print(f"\n✓ Papers that also have cs.CV or cs.CL category: {len(filtered_cv_cl)}")
print(f"  ({len(filtered_cv_cl)/len(sample_ai)*100:.1f}% of sample)")

print(f"\nSample papers with cs.CV or cs.CL (first 3):")
for i, doc_idx in enumerate(filtered_cv_cl[:3], 1):
    paper = bf.dataset.iloc[doc_idx]
    print(f"\n{i}. {paper['title'][:70]}...")
    print(f"   Primary: {paper['primary_category']}")
    print(f"   All categories: {', '.join(paper['categories'][:5])}")

# Demo 6: Full filtering pipeline
print("\n" + "="*70)
print("DEMO 6: Complete Filtering Pipeline")
print("="*70)

print("Scenario: Find papers from 2023-2024, cs.AI or cs.LG,")
print("          that also have cs.CV category")

# Step 1: Pre-filter
pre_filtered = bf.get_pre_filter_candidates(
    year_range=(2023.0, 2024.0),
    primary_categories=['cs.AI', 'cs.LG']
)
print(f"\nStep 1 - Pre-filter (year + primary category):")
print(f"  → {len(pre_filtered):,} candidates")

# Step 2: Simulate taking top 100 from search results
top_100 = pre_filtered[:100]
print(f"\nStep 2 - VSM/BM25F search (simulated):")
print(f"  → {len(top_100)} results")

# Step 3: Post-filter by categories
post_filtered = bf.filter_by_categories(top_100, ['cs.CV'])
print(f"\nStep 3 - Post-filter (categories):")
print(f"  → {len(post_filtered)} final results")

if len(post_filtered) > 0:
    print(f"\nFinal Results (first 3):")
    for i, doc_idx in enumerate(post_filtered[:3], 1):
        paper = bf.dataset.iloc[doc_idx]
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   Authors: {', '.join(paper['authors'][:2])}")
        print(f"   Year: {paper['year']}, Primary: {paper['primary_category']}")
        print(f"   Categories: {', '.join(paper['categories'][:4])}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY - Binary Model Capabilities")
print("="*70)
print(f"✓ Pre-filter fields:")
print(f"  - Year range: {bf.year_index['key'].nunique()} unique years")
print(f"  - Primary categories: {bf.primary_category_index['key'].nunique()} categories")
print(f"\n✓ Post-filter fields:")
print(f"  - Authors: {bf.authors_index['key'].nunique():,} unique authors")
print(f"  - Categories: {bf.categories_index['key'].nunique()} unique categories")
print(f"\n✓ Filtering performance:")
print(f"  - Filter lookup: O(1) via inverted index")
print(f"  - Set operations: O(n) where n = candidate size")
print(f"  - Memory: ~30 MB for all indexes")

print("\n" + "="*70)
print("DEMO SELESAI! ✓")
print("="*70)
print("\nBinary Model siap digunakan untuk:")
print("  1. Pre-filtering (year, primary_category) → VSM search space reduction")
print("  2. Post-filtering (authors, categories) → Result refinement")
print("  3. Integration dengan VSM/BM25F pipeline")
print("\nNext: Build VSM index dari full dataset untuk enable full integration")
