# Implementation Plan: Binary Model for ThesisFinder

## Context
Implement metadata filtering system (Binary Model) on top of existing VSM (TF-IDF) and BM25F search engines. Dataset has 380,879 papers with year (19.42% null), primary_category (4 unique values, 19.42% null), authors (numpy arrays), and categories (numpy arrays) fields.

## Design Decisions
1. **Index storage format**: Parquet files (fast loading, follows VSM pattern in src/vsm/indexer.py)
2. **Pre-filter vs post-filter split**: Year and primary_category are pre-filters (reduce search space before VSM) because they have high selectivity and low cardinality. Authors and categories are post-filters (filter VSM results) because they have very high cardinality (thousands of unique authors) and exact matching requirements.
3. **Filter logic**: Pre-filters use AND (intersection), post-filters use OR within each field type (ANY author matches, ANY category matches)
4. **Data format handling**: Authors and categories are numpy arrays in parquet, not string lists - iterate directly over array elements
5. **BM25F modification**: Add optional candidates parameter matching VSMSearcher.search() signature for consistency

## Implementation Plan

- [ ] 1. Create src/binary/config.py with configuration constants.
      Define BINARY_INDEX_DIR (e:\ThesisFinder\data\index\binary), DATASET_PATH, and field name constants.
      Files: src/binary/config.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.binary.config import BINARY_INDEX_DIR, DATASET_PATH; print(BINARY_INDEX_DIR, DATASET_PATH)"` — imports successfully and prints paths.

- [ ] 2. Create src/binary/indexer.py with BinaryIndexer class and index building logic.
      Build 4 inverted indexes as parquet files: year.parquet (key=year_value, doc_idx=row_index), primary_category.parquet, authors.parquet (one row per author per paper, normalized lowercase), categories.parquet (one row per category per paper). Authors and categories are numpy arrays in the dataset - iterate elements directly without ast.literal_eval. Skip nulls for year and primary_category. Save config.json with metadata (total_docs, unique_keys per field, non_null_docs, timestamp). Add CLI: `python -m src.binary.indexer`.
      Files: src/binary/indexer.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -m src.binary.indexer` then check e:\ThesisFinder\data\index\binary\ contains year.parquet, primary_category.parquet, authors.parquet, categories.parquet, config.json with correct stats.

- [ ] 3. Create src/binary/filter.py with BinaryFilter class.
      Load all 4 parquet indexes in __init__. Implement pre-filter methods: filter_by_year(year_min, year_max) returns np.array of doc indices, filter_by_primary_category(categories: list) with OR logic returns union, get_pre_filter_candidates(year_range, primary_categories) combines with AND (intersection). Implement post-filter methods: filter_by_authors(doc_ids, authors: list) and filter_by_categories(doc_ids, categories: list) use case-insensitive exact matching with ANY logic, apply_post_filters(doc_ids, authors, categories) applies both sequentially. Load full dataset parquet for accessing raw metadata during post-filtering.
      Files: src/binary/filter.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.binary.filter import BinaryFilter; bf = BinaryFilter(); candidates = bf.get_pre_filter_candidates(year_range=(2020, 2022), primary_categories=['cs.AI']); print(f'Returned {len(candidates)} candidates')"` — runs without errors and prints candidate count.

- [ ] 4. Create src/binary/__init__.py to export BinaryFilter and BinaryIndexer.
      Files: src/binary/__init__.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.binary import BinaryFilter, BinaryIndexer; print('Import successful')"` — imports work.

- [ ] 5. Modify src/algorithm/bm25f.py to add candidates parameter to search() method.
      Add candidates=None parameter to search(query, top_k=None, candidates=None). When candidates is not None, convert to set and only score documents in that set (filter during inverted index traversal). Ensure backward compatibility - existing calls without candidates work identically.
      Files: src/algorithm/bm25f.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.algorithm.bm25f import BM25F; docs = [{'title': 'test', 'abstract': 'hello'}, {'title': 'foo', 'abstract': 'world'}]; bm = BM25F(docs); print('Without candidates:', bm.search('test')); print('With candidates:', bm.search('test', candidates=[0]))"` — both calls work, second only returns doc 0.

- [ ] 6. Create src/pipeline/search_pipeline.py with SearchPipeline class.
      In __init__: instantiate VSMSearcher, BinaryFilter, load dataset parquet. Implement search(query, year_range, primary_categories, authors, all_categories, top_k=10): (1) get pre-filter candidates from BinaryFilter, (2) call VSM search with candidates, (3) apply post-filters to result doc_ids, (4) filter results list, (5) enrich with full metadata from dataset, (6) return list of dicts with metadata + score + rank. Handle edge cases (empty results, no filters, nulls).
      Files: src/pipeline/search_pipeline.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.pipeline.search_pipeline import SearchPipeline; sp = SearchPipeline(); results = sp.search('neural network', year_range=(2020, 2022), top_k=3); print(f'Found {len(results)} results'); [print(r['rank'], r['title'][:50], r['year']) for r in results]"` — runs and prints filtered results.

- [ ] 7. Add CLI to src/pipeline/search_pipeline.py using argparse.
      Add main() function and __main__ block. Support arguments: query (positional), --year-min, --year-max, --primary-categories (nargs='*'), --authors (nargs='*'), --all-categories (nargs='*'), --top (default 10). Format output with rank, score, title, authors, year, categories.
      Files: src/pipeline/search_pipeline.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -m src.pipeline.search_pipeline "machine learning" --year-min 2020 --year-max 2022 --primary-categories cs.AI cs.LG --top 5` — displays formatted search results.

- [ ] 8. Create src/pipeline/__init__.py to export SearchPipeline.
      Files: src/pipeline/__init__.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -c "from src.pipeline import SearchPipeline; print('Import successful')"` — imports work.

- [ ] 9. Create test/test_binary_model.py with comprehensive test suite.
      Create TestBinaryIndexer (test index building, multi-value fields, null handling), TestBinaryFilter (test all filter methods, AND/OR logic, case-insensitivity, edge cases), TestBM25FWithCandidates (test backward compatibility, filtering correctness), TestSearchPipeline (test full workflow, filter combinations, metadata enrichment, edge cases). Follow pattern in test/test_vsm.py using unittest.TestCase with pytest runner. Use small test dataset (10-20 papers) with controlled values.
      Files: test/test_binary_model.py
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -m pytest test/test_binary_model.py -v` — all tests pass.

- [ ] 10. Create docs/BINARY_MODEL.md documentation.
      Document architecture overview (pre-filter vs post-filter split, why each field is where), setup instructions (build indexes first with python -m src.binary.indexer), usage examples (Python API and CLI), filter behavior (AND/OR logic), performance notes (index build time, search latency targets), troubleshooting (null handling, case sensitivity).
      Files: docs/BINARY_MODEL.md
      Verify: Read the file and confirm it covers all sections clearly.

- [ ] 11. Run full test suite to verify integration.
      Run all tests including existing VSM tests to ensure no regressions.
      Files: None (verification only)
      Verify: `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -m pytest test/ -v` — all tests pass including test_vsm.py and test_binary_model.py.
