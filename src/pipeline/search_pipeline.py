"""
Search Pipeline - Orchestration lengkap filtering dan search.

Workflow:
1. Pre-filtering: year + primary_category -> candidates
2. VSM search dengan candidates
3. Post-filtering: authors + categories
4. Metadata enrichment
"""
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.binary import BinaryFilter
from src.vsm.searcher import VSMSearcher


class SearchPipeline:
    """
    Pipeline pencarian lengkap yang mengintegrasikan binary filtering dan VSM search.
    
    Workflow:
    - Pre-filtering (year_range, primary_categories) -> candidate doc indices
    - VSM search dengan candidates constraint
    - Post-filtering (authors, all_categories) -> filter results
    - Metadata enrichment dari dataset
    """
    
    def __init__(self, dataset_path: str = "e:/ThesisFinder/data/completed.parquet"):
        """
        Inisialisasi pipeline dengan semua komponen.
        
        Args:
            dataset_path: Path ke dataset parquet untuk metadata enrichment
        """
        # Load VSMSearcher
        self.vsm = VSMSearcher()
        
        # Load BinaryFilter
        self.bf = BinaryFilter()
        
        # Load dataset untuk metadata enrichment
        self.dataset = pd.read_parquet(dataset_path)
        
        print(f"SearchPipeline initialized:")
        print(f"  - VSM index: {self.vsm.n_docs:,} documents")
        print(f"  - Binary filter: {self.bf}")
        print(f"  - Dataset: {len(self.dataset):,} documents")
    
    def search(
        self,
        query: str,
        year_range: Optional[Tuple[float, float]] = None,
        primary_categories: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        all_categories: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Pencarian lengkap dengan filtering dan metadata enrichment.
        
        Workflow:
        1. Pre-filtering: year_range dan primary_categories -> candidates
        2. VSM search dengan candidates
        3. Post-filtering: authors dan all_categories
        4. Metadata enrichment
        
        Args:
            query: Query string untuk pencarian
            year_range: Tuple (year_min, year_max) untuk pre-filtering
            primary_categories: List of primary_category values untuk pre-filtering
            authors: List of author names untuk post-filtering
            all_categories: List of category names untuk post-filtering
            top_k: Jumlah maksimal hasil
        
        Returns:
            List of dicts dengan metadata + score + rank
        """
        # Step 1: Pre-filtering
        candidates = None
        if year_range is not None or primary_categories is not None:
            all_candidates = self.bf.get_pre_filter_candidates(
                year_range=year_range,
                primary_categories=primary_categories
            )
            
            # Filter candidates to only include doc indices that exist in VSM index
            # VSM index may be a subset of full dataset (e.g., first N documents)
            # We need to check which candidates are actually in the VSM index range
            candidates = all_candidates[all_candidates < self.vsm.n_docs]
            
            # Jika tidak ada candidates dari pre-filter yang ada di VSM index, return empty
            if len(candidates) == 0:
                print(f"Pre-filtering: {len(all_candidates):,} total candidates, but 0 within VSM index (0-{self.vsm.n_docs-1})")
                return []
            
            print(f"Pre-filtering: {len(candidates):,} candidates (from {len(all_candidates):,} total, filtered to VSM index 0-{self.vsm.n_docs-1})")
        
        # Step 2: VSM search
        # Jika candidates kosong (tidak ada pre-filter), VSM akan search semua dokumen
        vsm_results = self.vsm.search(
            query=query,
            candidates=candidates if candidates is not None and len(candidates) > 0 else None,
            top_k=top_k
        )
        
        # Jika tidak ada hasil dari VSM, return empty
        if not vsm_results:
            return []
        
        print(f"VSM search: {len(vsm_results)} results")
        
        # Step 3: Post-filtering
        doc_ids = np.array([r.doc_idx for r in vsm_results], dtype=np.int64)
        
        if authors is not None or all_categories is not None:
            filtered_doc_ids = self.bf.apply_post_filters(
                doc_ids=doc_ids,
                authors=authors,
                categories=all_categories
            )
            
            # Jika tidak ada yang lolos post-filter, return empty
            if len(filtered_doc_ids) == 0:
                return []
            
            print(f"Post-filtering: {len(filtered_doc_ids)} documents passed")
            
            # Filter hasil VSM berdasarkan doc_ids yang lolos post-filter
            filtered_doc_ids_set = set(filtered_doc_ids)
            vsm_results = [r for r in vsm_results if r.doc_idx in filtered_doc_ids_set]
        
        # Step 4: Metadata enrichment
        results = []
        for rank, result in enumerate(vsm_results, start=1):
            doc_idx = result.doc_idx
            doc_row = self.dataset.iloc[doc_idx]
            
            # Build enriched result
            enriched = {
                "rank": rank,
                "doc_idx": doc_idx,
                "score": result.score,
                "id": doc_row["id"],
                "title": doc_row["title"],
                "abstract": doc_row["abstract"] if pd.notna(doc_row["abstract"]) else None,
                "authors": doc_row["authors"].tolist() if isinstance(doc_row["authors"], np.ndarray) else [],
                "categories": doc_row["categories"].tolist() if isinstance(doc_row["categories"], np.ndarray) else [],
                "primary_category": doc_row["primary_category"] if pd.notna(doc_row["primary_category"]) else None,
                "year": int(doc_row["year"]) if pd.notna(doc_row["year"]) else None,
            }
            results.append(enriched)
        
        return results
    
    def __repr__(self):
        return f"SearchPipeline(vsm={self.vsm.n_docs} docs, bf={self.bf})"


# ===== CLI =====
def main():
    parser = argparse.ArgumentParser(
        description="ThesisFinder Search - Pencarian dengan filtering metadata"
    )
    
    # Query
    parser.add_argument("query", help="Query string untuk pencarian")
    
    # Pre-filters
    parser.add_argument("--year-min", type=float, default=None, help="Tahun minimum (inclusive)")
    parser.add_argument("--year-max", type=float, default=None, help="Tahun maksimum (inclusive)")
    parser.add_argument(
        "--primary-categories", 
        nargs="+", 
        default=None, 
        help="Primary categories untuk pre-filtering (OR logic)"
    )
    
    # Post-filters
    parser.add_argument(
        "--authors", 
        nargs="+", 
        default=None, 
        help="Author names untuk post-filtering (case-insensitive, ANY logic)"
    )
    parser.add_argument(
        "--all-categories", 
        nargs="+", 
        default=None, 
        help="Category names untuk post-filtering (case-insensitive, ANY logic)"
    )
    
    # Search parameters
    parser.add_argument("--top", type=int, default=10, help="Jumlah maksimal hasil (default: 10)")
    
    args = parser.parse_args()
    
    # Build year_range
    year_range = None
    if args.year_min is not None or args.year_max is not None:
        year_min = args.year_min if args.year_min is not None else 0
        year_max = args.year_max if args.year_max is not None else 9999
        year_range = (year_min, year_max)
    
    # Initialize pipeline
    print(f"Initializing pipeline...\n")
    pipeline = SearchPipeline()
    
    # Search
    print(f"\nSearching for: '{args.query}'")
    if year_range:
        print(f"  Year range: {year_range[0]} - {year_range[1]}")
    if args.primary_categories:
        print(f"  Primary categories: {args.primary_categories}")
    if args.authors:
        print(f"  Authors: {args.authors}")
    if args.all_categories:
        print(f"  Categories: {args.all_categories}")
    print()
    
    start_time = time.perf_counter()
    results = pipeline.search(
        query=args.query,
        year_range=year_range,
        primary_categories=args.primary_categories,
        authors=args.authors,
        all_categories=args.all_categories,
        top_k=args.top
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results in {elapsed_ms:.1f} ms")
    print(f"{'='*80}\n")
    
    for result in results:
        print(f"{result['rank']:>2}. [Score: {result['score']:.4f}] {result['title']}")
        print(f"    ID: {result['id']}")
        
        # Authors
        authors_str = ", ".join(result['authors']) if result['authors'] else "N/A"
        if len(authors_str) > 80:
            authors_str = authors_str[:77] + "..."
        print(f"    Authors: {authors_str}")
        
        # Year and primary category
        year_str = str(result['year']) if result['year'] else "N/A"
        primary_cat_str = result['primary_category'] if result['primary_category'] else "N/A"
        print(f"    Year: {year_str} | Primary Category: {primary_cat_str}")
        
        # Categories
        categories_str = ", ".join(result['categories']) if result['categories'] else "N/A"
        if len(categories_str) > 80:
            categories_str = categories_str[:77] + "..."
        print(f"    Categories: {categories_str}")
        
        # Abstract preview
        if result['abstract']:
            abstract_preview = result['abstract'][:150].replace('\n', ' ')
            if len(result['abstract']) > 150:
                abstract_preview += "..."
            print(f"    Abstract: {abstract_preview}")
        
        print()


if __name__ == "__main__":
    main()
