"""
Binary Model Indexer - Membangun inverted index untuk field year, primary_category, authors, categories.

Inverted index disimpan dalam format parquet dengan kolom [key, doc_idx].
- year.parquet: key = nilai year (float), doc_idx = row index di dataset
- primary_category.parquet: key = nilai primary_category (str), doc_idx = row index
- authors.parquet: key = nama author (str, lowercase normalized), doc_idx = row index
- categories.parquet: key = nama category (str, lowercase normalized), doc_idx = row index

Field year dan primary_category memiliki 73,966 nilai null yang di-skip.
Field authors dan categories adalah numpy arrays - setiap elemen array diindeks terpisah.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import BINARY_INDEX_DIR, DATASET_PATH, INDEXED_FIELDS


class BinaryIndexer:
    """
    Indexer untuk binary model yang membangun inverted index untuk filtering.
    """
    
    def __init__(self, dataset_path=DATASET_PATH, output_dir=BINARY_INDEX_DIR):
        """
        Inisialisasi indexer.
        
        Args:
            dataset_path: Path ke dataset parquet
            output_dir: Direktori output untuk menyimpan indeks
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.stats = {}
        
    def build_indexes(self, batch_size: int = 50_000, verbose: bool = True):
        """
        Membangun semua inverted index untuk field yang diindeks.
        
        Args:
            batch_size: Ukuran batch untuk membaca dataset
            verbose: Cetak progress ke console
        """
        if verbose:
            print(f"Membangun binary indexes dari {self.dataset_path}")
            print(f"Output: {self.output_dir}")
        
        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Baca dataset
        if verbose:
            print("Membaca dataset...")
        
        df = pd.read_parquet(self.dataset_path)
        total_docs = len(df)
        
        if verbose:
            print(f"Total dokumen: {total_docs:,}")
        
        # Build index untuk setiap field
        self._build_year_index(df, verbose)
        self._build_primary_category_index(df, verbose)
        self._build_authors_index(df, verbose)
        self._build_categories_index(df, verbose)
        
        # Simpan config
        config = {
            "total_docs": total_docs,
            "field_stats": self.stats,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.time() - t0, 1),
        }
        
        config_path = self.output_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        
        if verbose:
            print(f"\nSelesai dalam {config['build_seconds']}s -> {self.output_dir}")
            print(f"Config disimpan: {config_path}")
            
        return config
    
    def _build_year_index(self, df: pd.DataFrame, verbose: bool):
        """Build inverted index untuk field year."""
        if verbose:
            print("\nMembangun index: year")
        
        # Filter out null values
        valid_mask = df["year"].notna()
        valid_df = df[valid_mask].copy()
        
        # Build inverted index: {year_value: [doc_indices]}
        index_data = []
        for doc_idx, year_value in zip(valid_df.index, valid_df["year"]):
            index_data.append({"key": float(year_value), "doc_idx": int(doc_idx)})
        
        # Simpan sebagai parquet
        index_df = pd.DataFrame(index_data)
        output_path = self.output_dir / "year.parquet"
        index_df.to_parquet(output_path, engine="pyarrow", index=False)
        
        # Stats
        unique_years = index_df["key"].nunique()
        non_null_docs = len(index_df)
        
        self.stats["year"] = {
            "unique_keys": unique_years,
            "non_null_docs": non_null_docs,
            "total_entries": len(index_df),
        }
        
        if verbose:
            print(f"  - Unique years: {unique_years}")
            print(f"  - Non-null docs: {non_null_docs:,}")
            print(f"  - Total entries: {len(index_df):,}")
    
    def _build_primary_category_index(self, df: pd.DataFrame, verbose: bool):
        """Build inverted index untuk field primary_category."""
        if verbose:
            print("\nMembangun index: primary_category")
        
        # Filter out null values
        valid_mask = df["primary_category"].notna()
        valid_df = df[valid_mask].copy()
        
        # Build inverted index
        index_data = []
        for doc_idx, category in zip(valid_df.index, valid_df["primary_category"]):
            index_data.append({"key": str(category), "doc_idx": int(doc_idx)})
        
        # Simpan sebagai parquet
        index_df = pd.DataFrame(index_data)
        output_path = self.output_dir / "primary_category.parquet"
        index_df.to_parquet(output_path, engine="pyarrow", index=False)
        
        # Stats
        unique_categories = index_df["key"].nunique()
        non_null_docs = len(index_df)
        
        self.stats["primary_category"] = {
            "unique_keys": unique_categories,
            "non_null_docs": non_null_docs,
            "total_entries": len(index_df),
        }
        
        if verbose:
            print(f"  - Unique categories: {unique_categories}")
            print(f"  - Non-null docs: {non_null_docs:,}")
            print(f"  - Total entries: {len(index_df):,}")
    
    def _build_authors_index(self, df: pd.DataFrame, verbose: bool):
        """
        Build inverted index untuk field authors.
        Authors adalah numpy array - setiap author dalam array diindeks terpisah.
        Normalisasi: lowercase untuk case-insensitive matching.
        """
        if verbose:
            print("\nMembangun index: authors")
        
        # Build inverted index: satu entry per author per paper
        index_data = []
        docs_with_authors = 0
        
        for doc_idx, authors_array in enumerate(df["authors"]):
            # authors_array adalah numpy array
            if authors_array is not None and isinstance(authors_array, np.ndarray) and len(authors_array) > 0:
                docs_with_authors += 1
                for author in authors_array:
                    # Normalize: lowercase
                    author_normalized = str(author).lower().strip()
                    if author_normalized:
                        index_data.append({"key": author_normalized, "doc_idx": int(doc_idx)})
        
        # Simpan sebagai parquet
        index_df = pd.DataFrame(index_data)
        output_path = self.output_dir / "authors.parquet"
        index_df.to_parquet(output_path, engine="pyarrow", index=False)
        
        # Stats
        unique_authors = index_df["key"].nunique()
        
        self.stats["authors"] = {
            "unique_keys": unique_authors,
            "non_null_docs": docs_with_authors,
            "total_entries": len(index_df),
        }
        
        if verbose:
            print(f"  - Unique authors: {unique_authors:,}")
            print(f"  - Docs with authors: {docs_with_authors:,}")
            print(f"  - Total entries: {len(index_df):,}")
    
    def _build_categories_index(self, df: pd.DataFrame, verbose: bool):
        """
        Build inverted index untuk field categories.
        Categories adalah numpy array - setiap category dalam array diindeks terpisah.
        Normalisasi: lowercase untuk case-insensitive matching.
        """
        if verbose:
            print("\nMembangun index: categories")
        
        # Build inverted index: satu entry per category per paper
        index_data = []
        docs_with_categories = 0
        
        for doc_idx, categories_array in enumerate(df["categories"]):
            # categories_array adalah numpy array
            if categories_array is not None and isinstance(categories_array, np.ndarray) and len(categories_array) > 0:
                docs_with_categories += 1
                for category in categories_array:
                    # Normalize: lowercase
                    category_normalized = str(category).lower().strip()
                    if category_normalized:
                        index_data.append({"key": category_normalized, "doc_idx": int(doc_idx)})
        
        # Simpan sebagai parquet
        index_df = pd.DataFrame(index_data)
        output_path = self.output_dir / "categories.parquet"
        index_df.to_parquet(output_path, engine="pyarrow", index=False)
        
        # Stats
        unique_categories = index_df["key"].nunique()
        
        self.stats["categories"] = {
            "unique_keys": unique_categories,
            "non_null_docs": docs_with_categories,
            "total_entries": len(index_df),
        }
        
        if verbose:
            print(f"  - Unique categories: {unique_categories:,}")
            print(f"  - Docs with categories: {docs_with_categories:,}")
            print(f"  - Total entries: {len(index_df):,}")


def main():
    """CLI untuk membangun binary indexes."""
    ap = argparse.ArgumentParser(
        description="Bangun binary inverted indexes untuk year, primary_category, authors, categories"
    )
    ap.add_argument("--input", default=str(DATASET_PATH), help="Path dataset parquet")
    ap.add_argument("--output", default=str(BINARY_INDEX_DIR), help="Folder output indeks")
    ap.add_argument("--batch-size", type=int, default=50_000, help="Ukuran batch untuk membaca dataset")
    args = ap.parse_args()
    
    indexer = BinaryIndexer(dataset_path=args.input, output_dir=args.output)
    indexer.build_indexes(batch_size=args.batch_size, verbose=True)


if __name__ == "__main__":
    main()
