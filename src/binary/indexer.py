"""
Binary Model Indexer - Membangun inverted index untuk field year, primary_category, authors, categories.

Inverted index disimpan dalam format parquet dengan kolom [key, doc_idx].
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BINARY_INDEX_DIR, DATASET_PATH


class BinaryIndexer:
    """Indexer untuk binary model yang membangun inverted index untuk filtering."""

    def __init__(self, dataset_path=DATASET_PATH, output_dir=BINARY_INDEX_DIR):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.stats = {}

    def build_indexes(self, batch_size: int = 50_000, verbose: bool = True):
        if verbose:
            print(f"Membangun binary indexes dari {self.dataset_path}")
            print(f"Output: {self.output_dir}")

        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}

        if verbose:
            print("Membaca dataset...")

        df = pd.read_parquet(self.dataset_path).reset_index(drop=True)
        total_docs = len(df)

        if verbose:
            print(f"Total dokumen: {total_docs:,}")

        self._build_year_index(df, verbose)
        self._build_primary_category_index(df, verbose)
        self._build_authors_index(df, verbose)
        self._build_categories_index(df, verbose)

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
        if verbose:
            print("\nMembangun index: year")

        valid_mask = df["year"].notna()
        valid_df = df[valid_mask]
        index_df = pd.DataFrame({
            "key": valid_df["year"].astype(float).to_numpy(),
            "doc_idx": valid_df.index.to_numpy(dtype=np.int64),
        })
        index_df.to_parquet(self.output_dir / "year.parquet", engine="pyarrow", index=False)

        unique_years = int(index_df["key"].nunique())
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
        if verbose:
            print("\nMembangun index: primary_category")

        valid_mask = df["primary_category"].notna()
        valid_df = df[valid_mask]
        index_df = pd.DataFrame({
            "key": valid_df["primary_category"].astype(str).to_numpy(),
            "doc_idx": valid_df.index.to_numpy(dtype=np.int64),
        })
        index_df.to_parquet(self.output_dir / "primary_category.parquet", engine="pyarrow", index=False)

        unique_categories = int(index_df["key"].nunique())
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

    def _build_multi_value_index(self, series: pd.Series, field_name: str, verbose: bool):
        if verbose:
            print(f"\nMembangun index: {field_name}")

        keys: list[str] = []
        doc_indices: list[int] = []
        docs_non_empty = 0

        for doc_idx, values in enumerate(series):
            if values is not None and isinstance(values, (np.ndarray, list, tuple)) and len(values) > 0:
                added_for_doc = False
                for val in values:
                    norm = str(val).lower().strip()
                    if norm:
                        keys.append(norm)
                        doc_indices.append(doc_idx)
                        added_for_doc = True
                if added_for_doc:
                    docs_non_empty += 1

        index_df = pd.DataFrame({
            "key": pd.Series(keys, dtype="object"),
            "doc_idx": pd.Series(doc_indices, dtype=np.int64),
        })
        index_df.to_parquet(self.output_dir / f"{field_name}.parquet", engine="pyarrow", index=False)

        unique_keys = int(index_df["key"].nunique()) if len(index_df) > 0 else 0
        self.stats[field_name] = {
            "unique_keys": unique_keys,
            "non_null_docs": docs_non_empty,
            "total_entries": len(index_df),
        }

        if verbose:
            print(f"  - Unique {field_name}: {unique_keys:,}")
            print(f"  - Docs with {field_name}: {docs_non_empty:,}")
            print(f"  - Total entries: {len(index_df):,}")

    def _build_authors_index(self, df: pd.DataFrame, verbose: bool):
        self._build_multi_value_index(df["authors"], "authors", verbose)

    def _build_categories_index(self, df: pd.DataFrame, verbose: bool):
        self._build_multi_value_index(df["categories"], "categories", verbose)


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
