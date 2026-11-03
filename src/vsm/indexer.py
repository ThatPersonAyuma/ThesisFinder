import argparse
import json
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .tfidf import (DATASET_PATH, FIELD_ANALYZERS, FIELDS, INDEX_DIR, INDEX_FORMAT_VERSION,
                    row_ids_from_indptr, weight_matrix_data)


READ_COLUMNS = ["id", "title", "abstract"]


def iter_papers(path, batch_size: int = 20_000, limit: int | None = None) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(path)
    missing = set(READ_COLUMNS) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {sorted(missing)}")
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=READ_COLUMNS):
        df = batch.to_pandas()
        if limit is not None and seen + len(df) > limit:
            df = df.iloc[: limit - seen]
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        df["abstract"] = df["abstract"].fillna("").astype(str)
        yield df.reset_index(drop=True)
        seen += len(df)
        if limit is not None and seen >= limit:
            break


def build_index(batches, out_dir=INDEX_DIR, fields=FIELDS, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vocab: dict[str, int] = {}
    # Per field: tiga buffer ringkas (CSR mentah berisi jumlah kemunculan term)
    buf = {f: {"indices": array("i"), "counts": array("i"), "indptr": array("q", [0])} for f in fields}
    meta_parts = []
    n_docs = 0

    for batch in batches:
        for f in fields:
            analyzer = FIELD_ANALYZERS[f]
            b = buf[f]
            for text in batch[f].tolist():
                for term, c in Counter(analyzer(text)).items():
                    tid = vocab.get(term)
                    if tid is None:
                        tid = vocab[term] = len(vocab)
                    b["indices"].append(tid)
                    b["counts"].append(c)
                b["indptr"].append(len(b["indices"]))
        meta_parts.append(batch[["id", "title"]])
        n_docs += len(batch)
        if verbose:
            print(f"  diproses {n_docs:,} dokumen | vocab {len(vocab):,} | {time.time() - t0:.0f}s", flush=True)

    n_terms = len(vocab)
    field_stats = {}
    for f in fields:
        b = buf[f]
        indices = np.frombuffer(b["indices"], dtype=np.int32).astype(np.int32)
        counts = np.frombuffer(b["counts"], dtype=np.int32)
        indptr = np.frombuffer(b["indptr"], dtype=np.int64)
        data, idf, n_nonempty = weight_matrix_data(indices, counts, indptr, n_terms)
        
        pd.DataFrame({
            "doc": row_ids_from_indptr(indptr).astype(np.int32),
            "term": indices.astype(np.int32),
            "w": data.astype(np.float32),
        }).to_parquet(out_dir / f"docvec_{f}.parquet", engine="pyarrow", index=False)
        np.save(out_dir / f"idf_{f}.npy", idf.astype(np.float32))
        np.save(out_dir / f"df_{f}.npy", np.bincount(indices, minlength=n_terms).astype(np.int32))
        field_stats[f] = {"docs_nonempty": n_nonempty, "nnz": int(len(data))}
        if verbose:
            print(f"  field '{f}': {n_nonempty:,} dokumen terisi, {len(data):,} nilai non-nol", flush=True)
        del indices, counts, indptr, data

    ids_terms = [None] * n_terms
    for term, tid in vocab.items():
        ids_terms[tid] = term
    (out_dir / "vocab.json").write_text(json.dumps(ids_terms), encoding="utf-8")
    # meta: urutan baris = nomor dokumen (doc_idx). Hanya id (kunci ke dataset) dan title (untuk tampilan).
    pd.concat(meta_parts, ignore_index=True).to_parquet(out_dir / "meta.parquet", engine="pyarrow")

    config = {
        "format_version": INDEX_FORMAT_VERSION,
        "n_docs": n_docs,
        "n_terms": n_terms,
        "fields": list(fields),
        "field_stats": field_stats,
        "weighting": "tf=1+ln(count); idf=ln((N+1)/(df+1))+1; L2-normalized rows; cosine",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if verbose:
        print(f"Selesai dalam {config['build_seconds']}s -> {out_dir}")
    return config


def build_index_from_parquet(path, out_dir=INDEX_DIR, limit: int | None = None,
                             batch_size: int = 20_000, verbose: bool = True) -> dict:
    return build_index(iter_papers(path, batch_size=batch_size, limit=limit), out_dir, verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description="Bangun indeks TF-IDF pre-compute (VSM: title + abstract)")
    ap.add_argument("--input", default=str(DATASET_PATH), help="path dataset parquet")
    ap.add_argument("--output", default=str(INDEX_DIR), help="folder keluaran indeks")
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah dokumen (untuk uji cepat)")
    ap.add_argument("--batch-size", type=int, default=20_000)
    args = ap.parse_args()
    build_index_from_parquet(args.input, args.output, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()