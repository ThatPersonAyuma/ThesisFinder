"""
Konfigurasi untuk Binary Model Indexer dan Filter.
"""
from pathlib import Path

# Direktori penyimpanan indeks binary (parquet files)
BINARY_INDEX_DIR = Path("e:/ThesisFinder/data/index/binary")

# Path ke dataset lengkap
DATASET_PATH = Path("e:/ThesisFinder/data/completed.parquet")

# Nama field yang diindeks untuk pre-filtering (year dan primary_category)
PRE_FILTER_FIELDS = ["year", "primary_category"]

# Nama field yang diindeks untuk post-filtering (authors dan categories)
POST_FILTER_FIELDS = ["authors", "categories"]

# Semua field yang diindeks
INDEXED_FIELDS = PRE_FILTER_FIELDS + POST_FILTER_FIELDS
