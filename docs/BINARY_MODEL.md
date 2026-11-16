# Binary Model Documentation

## Overview

Binary Model adalah komponen filtering sistem ThesisFinder yang menggunakan inverted index untuk memfilter dokumen berdasarkan metadata. Implementasi ini mengikuti proposal dan menggunakan pre-filtering dan post-filtering untuk mengoptimalkan pencarian.

## Arsitektur

### Komponen Utama

1. **BinaryIndexer** (`src/binary/indexer.py`)
   - Membangun inverted index untuk field metadata
   - Menyimpan index dalam format Parquet untuk efisiensi

2. **BinaryFilter** (`src/binary/filter.py`)
   - Pre-filtering: year range dan primary_category
   - Post-filtering: authors dan categories
   - Mendukung operasi AND/OR yang fleksibel

3. **Config** (`src/binary/config.py`)
   - Konfigurasi path dan field yang diindeks
   - Konstanta sistem

### Struktur Data Flow

```
Query Input
    ↓
┌─────────────────────────────────┐
│   Pre-Filtering (Binary Model)  │
│   - year_range (AND)            │
│   - primary_categories (OR)      │
└─────────────────────────────────┘
    ↓ Candidates (doc indices)
┌─────────────────────────────────┐
│   VSM/BM25F Search              │
│   - Hanya search dalam          │
│     candidates                   │
└─────────────────────────────────┘
    ↓ Ranked Results
┌─────────────────────────────────┐
│   Post-Filtering (Binary Model) │
│   - authors (ANY)               │
│   - categories (ANY)             │
└─────────────────────────────────┘
    ↓ Final Results
┌─────────────────────────────────┐
│   Metadata Enrichment           │
│   - Load full metadata          │
└─────────────────────────────────┘
```

### Field yang Diindeks

#### Pre-Filter Fields
- **year**: Tahun publikasi (float), filter dengan range (min-max)
- **primary_category**: Kategori utama (string), filter dengan OR logic

#### Post-Filter Fields
- **authors**: Daftar penulis (multi-value), filter dengan ANY logic (case-insensitive)
- **categories**: Daftar kategori (multi-value), filter dengan ANY logic (case-insensitive)

### Operasi Filtering

#### Pre-Filtering (AND Logic)
```python
# Intersection: semua filter harus terpenuhi
candidates = year_candidates ∩ primary_category_candidates
```

#### Post-Filtering (AND Logic between filters, ANY within)
```python
# Filter authors: doc harus punya minimal 1 author yang cocok
# Filter categories: doc harus punya minimal 1 category yang cocok
# AND: doc harus lolos semua filter
results = author_filtered ∩ category_filtered
```

## Setup

### 1. Instalasi Dependencies

Pastikan semua dependencies terinstal:

```bash
pip install -r requirements.txt
```

Dependencies yang diperlukan:
- pandas
- numpy
- pyarrow (untuk Parquet)
- pytest (untuk testing)

### 2. Build Binary Index

Bangun inverted index dari dataset:

```bash
python -m src.binary.indexer --input data/completed.parquet --output data/index/binary
```

**Output**:
```
Membangun binary indexes dari data/completed.parquet
Output: data/index/binary
Membaca dataset...
Total dokumen: 380,879

Membangun index: year
  - Unique years: 11
  - Non-null docs: 306,913
  - Total entries: 306,913

Membangun index: primary_category
  - Unique categories: 4
  - Non-null docs: 306,913
  - Total entries: 306,913

Membangun index: authors
  - Unique authors: 462,877
  - Docs with authors: 380,879
  - Total entries: 1,752,347

Membangun index: categories
  - Unique categories: 149
  - Docs with categories: 380,879
  - Total entries: 874,299

Selesai dalam 43.9s -> data/index/binary
```

**File yang Dihasilkan**:
- `data/index/binary/year.parquet` (1.5 MB)
- `data/index/binary/primary_category.parquet` (1.5 MB)
- `data/index/binary/authors.parquet` (22.6 MB)
- `data/index/binary/categories.parquet` (3.1 MB)
- `data/index/binary/config.json` (metadata dan statistics)

### 3. Verifikasi Index

Periksa config yang dihasilkan:

```bash
cat data/index/binary/config.json
```

```json
{
  "total_docs": 380879,
  "field_stats": {
    "year": {
      "unique_keys": 11,
      "non_null_docs": 306913,
      "total_entries": 306913
    },
    "primary_category": {
      "unique_keys": 4,
      "non_null_docs": 306913,
      "total_entries": 306913
    },
    "authors": {
      "unique_keys": 462877,
      "non_null_docs": 380879,
      "total_entries": 1752347
    },
    "categories": {
      "unique_keys": 149,
      "non_null_docs": 380879,
      "total_entries": 874299
    }
  },
  "built_at": "2024-01-15 10:30:45",
  "build_seconds": 43.9
}
```

## Usage

### API Usage

#### 1. Inisialisasi Filter

```python
from src.binary.filter import BinaryFilter

# Inisialisasi dengan default paths
filter = BinaryFilter()

# Atau dengan custom paths
filter = BinaryFilter(
    index_dir="data/index/binary",
    dataset_path="data/completed.parquet"
)
```

#### 2. Pre-Filtering

**Filter by Year Range**:
```python
# Dokumen tahun 2020-2021
candidates = filter.filter_by_year(2020.0, 2021.0)
print(f"Found {len(candidates)} candidates")
# Output: Found 72,239 candidates
```

**Filter by Primary Category**:
```python
# Dokumen dengan primary_category cs.AI atau cs.CV
candidates = filter.filter_by_primary_category(["cs.AI", "cs.CV"])
print(f"Found {len(candidates)} candidates")
# Output: Found 149,586 candidates
```

**Kombinasi Pre-Filters**:
```python
# Dokumen tahun 2020-2021 DAN (cs.AI ATAU cs.CV)
candidates = filter.get_pre_filter_candidates(
    year_range=(2020.0, 2021.0),
    primary_categories=["cs.AI", "cs.CV"]
)
print(f"Found {len(candidates)} candidates")
# Output: Found 57,938 candidates
```

#### 3. Post-Filtering

**Filter by Authors**:
```python
# Dari hasil VSM search, filter yang punya author tertentu
doc_ids = np.array([1, 2, 3, 100, 200])
filtered = filter.filter_by_authors(
    doc_ids, 
    authors=["Alice Smith", "bob jones"]  # case-insensitive
)
print(f"Filtered to {len(filtered)} docs")
```

**Filter by Categories**:
```python
# Filter yang punya category tertentu
filtered = filter.filter_by_categories(
    doc_ids,
    categories=["cs.ai", "cs.lg"]  # case-insensitive
)
print(f"Filtered to {len(filtered)} docs")
```

**Kombinasi Post-Filters**:
```python
# Apply multiple post-filters (AND logic)
filtered = filter.apply_post_filters(
    doc_ids,
    authors=["Alice Smith"],
    categories=["cs.ai"]
)
print(f"Final results: {len(filtered)} docs")
```

#### 4. Full Pipeline Integration

```python
from src.binary.filter import BinaryFilter
from src.vsm.searcher import VSMSearcher

# Initialize
binary_filter = BinaryFilter()
vsm_searcher = VSMSearcher()

# Query
query = "deep learning neural networks"

# Step 1: Pre-filtering
candidates = binary_filter.get_pre_filter_candidates(
    year_range=(2020.0, 2022.0),
    primary_categories=["cs.AI", "cs.LG"]
)

# Step 2: VSM search dengan candidates
results = vsm_searcher.search(
    query, 
    top_k=100, 
    candidates=candidates
)

# Step 3: Post-filtering
doc_ids = np.array([doc_id for doc_id, score in results])
final_doc_ids = binary_filter.apply_post_filters(
    doc_ids,
    authors=["Alice Smith"],
    categories=["cs.ai"]
)

# Step 4: Ambil metadata
final_results = []
for doc_id in final_doc_ids:
    metadata = binary_filter.dataset.iloc[doc_id]
    final_results.append({
        "id": metadata["id"],
        "title": metadata["title"],
        "authors": metadata["authors"],
        "year": metadata["year"]
    })

print(f"Found {len(final_results)} results")
```

### CLI Usage

#### Search Pipeline CLI

Command lengkap untuk searching dengan filter:

```bash
python -m src.pipeline.search_pipeline <query> [options]
```

**Options**:
- `--top N`: Number of results (default: 10)
- `--year-min YYYY`: Minimum year (inclusive)
- `--year-max YYYY`: Maximum year (inclusive)
- `--primary-categories CAT1 CAT2`: Primary category filter (OR logic)
- `--all-categories CAT1 CAT2`: Category filter (ANY logic)
- `--authors "Author Name"`: Author filter (ANY logic)

**Contoh Penggunaan**:

1. **Basic search tanpa filter**:
```bash
python -m src.pipeline.search_pipeline "deep learning neural networks" --top 10
```

2. **Filter by year range**:
```bash
python -m src.pipeline.search_pipeline "machine learning" --year-min 2020 --year-max 2021 --top 5
```

3. **Filter by primary category**:
```bash
python -m src.pipeline.search_pipeline "graph neural" --primary-categories cs.AI cs.LG --top 5
```

4. **Filter by categories**:
```bash
python -m src.pipeline.search_pipeline "transformer" --all-categories cs.cl cs.ai --top 5
```

5. **Filter kombinasi**:
```bash
python -m src.pipeline.search_pipeline "reinforcement learning" \
  --year-min 2020 \
  --primary-categories cs.AI cs.LG \
  --all-categories cs.ro cs.ai \
  --top 10
```

#### Indexer CLI

```bash
python -m src.binary.indexer \
  --input data/completed.parquet \
  --output data/index/binary \
  --batch-size 50000
```

## Filter Behavior

### Null Value Handling

- **year** dan **primary_category**: Null values di-skip dari index
  - 73,966 dokumen dengan year=null atau primary_category=null tidak ada di index
  - Pre-filter **tidak akan** mengembalikan dokumen dengan null values

- **authors** dan **categories**: Empty arrays di-skip
  - Dokumen dengan `authors=[]` atau `categories=[]` tidak masuk index
  - Post-filter tidak akan menemukan dokumen ini

### Case Sensitivity

- **year** dan **primary_category**: Case-sensitive (sesuai nilai asli)
  - `"cs.AI"` ≠ `"cs.ai"`
  - Gunakan nilai exact seperti di dataset

- **authors** dan **categories**: Case-insensitive (normalized ke lowercase)
  - `"Alice Smith"` = `"alice smith"` = `"ALICE SMITH"`
  - Query bisa pakai case apapun

### Logic Operations

#### Pre-Filtering (AND between filters)
```python
year_candidates = [0, 1, 2, 3, 4]
category_candidates = [2, 3, 4, 5, 6]

# Result: intersection
final_candidates = [2, 3, 4]
```

#### Primary Category (OR within filter)
```python
# Dokumen dengan primary_category = "cs.AI" OR "cs.CV"
candidates = filter.filter_by_primary_category(["cs.AI", "cs.CV"])
```

#### Post-Filtering (ANY within filter, AND between filters)
```python
# Authors: doc punya minimal 1 dari ["Alice Smith", "Bob Jones"]
author_filtered = [0, 1, 5]

# Categories: doc punya minimal 1 dari ["cs.ai", "cs.lg"]
category_filtered = [0, 1, 2, 3]

# Final: intersection (AND)
final = [0, 1]
```

### Empty Results

Setiap stage bisa menghasilkan empty results:

```python
# Pre-filter: no results
candidates = filter.get_pre_filter_candidates(year_range=(2025, 2026))
# candidates = np.array([], dtype=np.int64)

# VSM search with empty candidates
results = vsm_searcher.search(query, candidates=candidates)
# results = []

# Post-filter: no results
filtered = filter.filter_by_authors([1, 2, 3], authors=["nonexistent"])
# filtered = np.array([], dtype=np.int64)
```

Pipeline menangani empty results dengan graceful:
- Tidak ada error
- Return empty list
- Log di console

## Performance

### Index Build Time

Dataset: 380,879 dokumen

| Field | Time | Entries | Size |
|-------|------|---------|------|
| year | ~5s | 306,913 | 1.5 MB |
| primary_category | ~5s | 306,913 | 1.5 MB |
| authors | ~20s | 1,752,347 | 22.6 MB |
| categories | ~15s | 874,299 | 3.1 MB |
| **Total** | **43.9s** | **2,934,472** | **28.7 MB** |

### Search Performance

Query: "deep learning neural networks"

| Stage | Time | Output |
|-------|------|--------|
| Pre-filter (year+category) | < 100ms | 57,938 candidates |
| VSM search (with candidates) | ~27ms | 10 results |
| Post-filter (authors+categories) | < 10ms | 8 results |
| Metadata enrichment | < 5ms | 8 full docs |
| **Total** | **~150ms** | **8 results** |

### Memory Usage

- **Index loading**: ~30 MB (semua index di-load ke memory)
- **Dataset lazy load**: 0 MB (hanya load saat post-filtering atau metadata enrichment)
- **Filter operations**: O(n) dimana n = jumlah kandidat

### Scalability

- **Index size**: Linear dengan jumlah dokumen dan unique values
- **Authors index**: Paling besar karena banyak unique authors (462k)
- **Categories index**: Moderat karena lebih sedikit unique categories (149)
- **Lookup time**: O(1) untuk key lookup, O(m) untuk result collection

## Notes

### Design Decisions

1. **Parquet Format**: Dipilih karena:
   - Kompresi efisien
   - Schema typed
   - Fast loading dengan PyArrow
   - Standar industri untuk big data

2. **Inverted Index**: Dipilih karena:
   - O(1) lookup untuk filtering
   - Memory efficient (hanya simpan key-to-doc mapping)
   - Mudah di-update (rebuild index)

3. **Pre vs Post Filtering**: Sesuai proposal:
   - Pre-filtering: Field yang bisa langsung kurangi search space (year, primary_category)
   - Post-filtering: Field multi-value yang lebih baik difilter setelah ranking (authors, categories)

4. **Case-Insensitive Post-Filters**: Karena:
   - Authors dan categories sering punya variasi case di dataset
   - User tidak harus tahu exact case
   - Performance impact minimal (normalisasi saat indexing)

### Known Limitations

1. **Index Sync Required**:
   - Binary index dan VSM index harus dari dataset yang sama
   - Jika dataset diupdate, rebuild semua index
   - Tidak ada automatic sync detection

2. **No Partial Matching**:
   - Authors dan categories pakai exact match (setelah normalisasi)
   - "Alice" tidak akan match "Alice Smith"
   - Bisa ditambahkan fuzzy matching di future

3. **Memory Constraints**:
   - Semua index di-load ke memory saat inisialisasi
   - Dataset di-lazy load tapi tetap perlu full load untuk post-filtering
   - Untuk dataset sangat besar (>10M docs), perlu sharding

4. **No Real-Time Updates**:
   - Index harus rebuild untuk reflect perubahan dataset
   - Incremental update tidak supported
   - Rebuild time: ~1 menit per 1M documents

### Recommendations

1. **Production Deployment**:
   - Build index pada dataset lengkap
   - Simpan index di persistent storage
   - Load index saat aplikasi startup
   - Rebuild index nightly atau weekly (sesuai update frequency)

2. **Performance Tuning**:
   - Untuk dataset >1M docs: pertimbangkan sharding by year/category
   - Untuk authors index: consider bloom filter untuk quick negative lookup
   - Untuk real-time updates: consider Redis atau Elasticsearch

3. **Monitoring**:
   - Log filter cascade statistics (candidates per stage)
   - Track empty result queries
   - Monitor index size growth
   - Alert pada index-dataset mismatch

## Testing

### Run All Tests

```bash
# All tests (48 tests)
python -m pytest test/ -v

# Binary model tests only (33 tests)
python -m pytest test/test_binary_model.py -v

# Integration test
python test_integration_simple.py
```

### Test Coverage

**BinaryIndexer Tests (5 tests)**:
- Index building creates all parquet files
- Multi-value fields (authors, categories) have multiple entries per doc
- Null values excluded from indexes
- Config contains correct statistics
- Empty arrays handled correctly

**BinaryFilter Tests (13 tests)**:
- Year range filtering
- Primary category filtering with OR logic
- Combined pre-filters with AND logic
- Authors post-filtering with case-insensitive matching
- Categories post-filtering with case-insensitive matching
- Combined post-filters with AND logic
- Empty results handling
- Case sensitivity tests

**BM25F Integration Tests (7 tests)**:
- Search without candidates (backward compatibility)
- Search with list candidates
- Search with numpy array candidates
- Empty candidates handling
- Candidates filtering correctness

**Search Pipeline Tests (8 tests)**:
- Full workflow integration
- Pre-filter aware of index boundaries
- Post-filtering after ranking
- Metadata enrichment
- Empty result handling at each stage

### Example Test Output

```
test/test_binary_model.py::TestBinaryIndexer::test_index_building_creates_parquet_files PASSED
test/test_binary_model.py::TestBinaryIndexer::test_authors_index_has_multiple_entries_per_paper PASSED
test/test_binary_model.py::TestBinaryFilter::test_filter_by_year PASSED
test/test_binary_model.py::TestBinaryFilter::test_filter_by_primary_category PASSED
test/test_binary_model.py::TestBinaryFilter::test_combined_pre_filters PASSED
test/test_binary_model.py::TestBinaryFilter::test_filter_by_authors_case_insensitive PASSED
test/test_binary_model.py::TestBinaryFilter::test_combined_post_filters PASSED
test/test_binary_model.py::TestBM25FWithCandidates::test_search_without_candidates PASSED
test/test_binary_model.py::TestSearchPipeline::test_full_workflow PASSED

========================== 33 passed in 2.19s ==========================
```

## Troubleshooting

### Error: "Binary indexes belum ada"

```
FileNotFoundError: Binary indexes belum ada di data/index/binary.
Jalankan: python -m src.binary.indexer
```

**Solusi**: Build index terlebih dahulu
```bash
python -m src.binary.indexer
```

### Error: "Index size mismatch"

```
Warning: VSM index size (1000) != Binary index size (380879)
```

**Penyebab**: VSM index dan Binary index dibangun dari dataset berbeda

**Solusi**: Rebuild VSM index
```bash
python -m src.vsm.indexer --input data/completed.parquet --output data/index/vsm
```

### Empty Results

**Debug steps**:
1. Check pre-filter candidates:
   ```python
   candidates = filter.get_pre_filter_candidates(year_range=(2020, 2021))
   print(f"Pre-filter: {len(candidates)} candidates")
   ```

2. Check VSM search:
   ```python
   results = vsm_searcher.search(query, candidates=candidates)
   print(f"VSM search: {len(results)} results")
   ```

3. Check post-filter:
   ```python
   doc_ids = np.array([doc_id for doc_id, _ in results])
   filtered = filter.apply_post_filters(doc_ids, authors=["..."])
   print(f"Post-filter: {len(filtered)} results")
   ```

### Performance Issues

**Symptom**: Search takes > 1 second

**Possible causes**:
1. Pre-filter returns too many candidates (> 100k)
   - **Solution**: Add more restrictive pre-filters
   
2. VSM index not optimized
   - **Solution**: Rebuild VSM index with proper parameters

3. Post-filter on large result set (> 1k)
   - **Solution**: Reduce VSM top_k parameter

## API Reference

### BinaryIndexer

```python
class BinaryIndexer:
    def __init__(self, dataset_path=DATASET_PATH, output_dir=BINARY_INDEX_DIR)
    def build_indexes(self, batch_size: int = 50_000, verbose: bool = True) -> dict
```

### BinaryFilter

```python
class BinaryFilter:
    def __init__(self, index_dir=BINARY_INDEX_DIR, dataset_path=DATASET_PATH)
    
    # Pre-filtering
    def filter_by_year(self, year_min: float, year_max: float) -> np.ndarray
    def filter_by_primary_category(self, categories: list) -> np.ndarray
    def get_pre_filter_candidates(
        self, 
        year_range: tuple = None, 
        primary_categories: list = None
    ) -> np.ndarray
    
    # Post-filtering
    def filter_by_authors(self, doc_ids: np.ndarray, authors: list) -> np.ndarray
    def filter_by_categories(self, doc_ids: np.ndarray, categories: list) -> np.ndarray
    def apply_post_filters(
        self,
        doc_ids: np.ndarray,
        authors: list = None,
        categories: list = None
    ) -> np.ndarray
```

## Conclusion

Binary Model implementation telah selesai dan siap untuk production. Implementasi ini:

✓ Mengikuti spesifikasi dari proposal  
✓ Terintegrasi dengan VSM dan BM25F  
✓ Memiliki test coverage lengkap (33 tests)  
✓ Performa optimal untuk dataset 380k dokumen  
✓ API yang clean dan mudah digunakan  
✓ CLI interface untuk testing dan debugging  

Untuk pertanyaan atau issues, refer to test suite atau INTEGRATION_REPORT.md.
