# ThesisFinder

Sistem temu kembali jurnal ilmiah (380.879 paper arXiv) yang menggabungkan **Binary Model** (pre/post-filtering metadata), **VSM TF-IDF** (first-stage retrieval), dan **Okapi BM25F** (multi-field re-ranking).

## Setup Setelah Clone

Pastikan Git LFS terpasang agar dataset `.parquet` dan CSV kosakata terunduh penuh:

```bash
git clone -b coba_gabung https://github.com/ThatPersonAyuma/ThesisFinder.git
cd ThesisFinder
git lfs pull
pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Pencarian Lengkap (`Binary + VSM + BM25F`)
Jika indeks belum tersedia setelah clone, perintah CLI `search_pipeline` akan otomatis membangun indeks VSM dan Binary pada pemakaian pertama:

```bash
# Mode hybrid (default: Binary pre-filter -> VSM -> BM25F re-ranking -> Binary post-filter)
python -m src.pipeline.search_pipeline "graph neural network" --mode hybrid --top 5

# Dengan filter metadata
python -m src.pipeline.search_pipeline "graph neural network" --year-min 2020 --year-max 2024 --primary-categories cs.AI cs.LG --top 5
python -m src.pipeline.search_pipeline "transformer attention" --mode vsm --all-categories cs.cl --top 5
```

Atau bangun indeks secara eksplisit terlebih dahulu:

```bash
python -m src.pipeline.setup
```

### 2. Modul VSM Mandiri
```bash
# Pencarian VSM (title + abstract)
python -m src.vsm.searcher "graph neural network" --top 5

# Demo tokenisasi & vektorisasi BoW / TF-IDF
python coba_vektor.py

# Bangun ulang kosakata bersih (data/vsm/vocab_clean.csv)
python -m src.vsm.cleaning
```

### 3. Menjalankan Test
```bash
# Seluruh unit & integration test (VSM, Binary, BM25F, Pipeline)
python -m unittest discover -s test -v

# Unit test BM25F standalone
python src/algorithm/bm25-test.py -v

# Verifikasi konsistensi indeks VSM & Binary
python test_integration_simple.py
```
