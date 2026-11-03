import argparse
import json
import re
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from .tfidf import (DATASET_PATH, FIELD_ANALYZERS, FIELDS, INDEX_DIR, INDEX_FORMAT_VERSION,
                    weight_matrix_data)

# DATASET 
TARGET_COLUMNS = ["id", "title", "abstract", "authors", "categories", "primary_category","published", "year", "month", "pdf", "keywords"]

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", re.I)
_KW_PREFIX_RE = re.compile(r"^\s*(?:index\s*terms?|key\s*words?)\s*[:\-\u2014\u2013.\u00b7]*\s*", re.I)
_KW_SPLIT_RE = re.compile(r"[,;\u00b7\u2022|]")


def clean_keywords(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(k).strip() for k in raw if str(k).strip()]
    text = _KW_PREFIX_RE.sub("", str(raw), count=1)
    items = [p.strip(" .\t\r\n") for p in _KW_SPLIT_RE.split(text)]
    return [p for p in items if len(p) > 1]


def _year_month_from_id(ids: pd.Series):
    m = ids.str.extract(_ARXIV_ID_RE)
    yy = pd.to_numeric(m[0], errors="coerce")
    mm = pd.to_numeric(m[1], errors="coerce")
    return (2000 + yy).astype("Int64"), mm.astype("Int64")


def normalize_batch(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"].astype(str)
    out["title"] = df["title"].fillna("").astype(str)
    out["abstract"] = df["abstract"].fillna("").astype(str)
    out["authors"] = df["authors"].apply(lambda a: list(a) if a is not None else [])
    out["categories"] = df["categories"].apply(lambda c: list(c) if c is not None else [])

    if "primary_category" in df.columns:
        out["primary_category"] = df["primary_category"]
    else:
        out["primary_category"] = out["categories"].apply(lambda c: c[0] if len(c) else None)

    if "published" in df.columns:
        out["published"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        out["published"] = pd.NaT

    id_year, id_month = _year_month_from_id(out["id"])
    if "year" in df.columns:
        out["year"] = df["year"].astype("Int64")
    else:
        out["year"] = out["published"].dt.year.astype("Int64").fillna(id_year)
    if "month" in df.columns:
        out["month"] = df["month"].astype("Int64")
    else:
        out["month"] = out["published"].dt.month.astype("Int64").fillna(id_month)

    if "pdf" in df.columns:
        out["pdf"] = df["pdf"]
    else:
        out["pdf"] = out["id"].str.replace("/abs/", "/pdf/", regex=False)

    out["keywords"] = df["keywords"].apply(clean_keywords) if "keywords" in df.columns else [[] for _ in range(len(out))]
    return out[TARGET_COLUMNS].reset_index(drop=True)


def iter_papers(path, batch_size: int = 20_000, limit: int | None = None) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    columns = [c for c in TARGET_COLUMNS + ["update_date"] if c in available]
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        if limit is not None and seen + len(df) > limit:
            df = df.iloc[: limit - seen]
        yield normalize_batch(df)
        seen += len(df)
        if limit is not None and seen >= limit:
            break


# PEMBANGUN INDEKS 
def _field_text(df: pd.DataFrame, field: str) -> list[str]:
    if field in ("authors", "keywords"):
        return [" ".join(v) for v in df[field]]
    return df[field].tolist()


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
            for text in _field_text(batch, f):
                for term, c in Counter(analyzer(text)).items():
                    tid = vocab.get(term)
                    if tid is None:
                        tid = vocab[term] = len(vocab)
                    b["indices"].append(tid)
                    b["counts"].append(c)
                b["indptr"].append(len(b["indices"]))
        meta_parts.append(batch.drop(columns=["abstract"]))
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
        csr = sp.csr_matrix((data.astype(np.float32), indices, indptr), shape=(n_docs, n_terms))
        csc = csr.tocsc()
        sp.save_npz(out_dir / f"tfidf_{f}.npz", csc, compressed=False)
        np.save(out_dir / f"idf_{f}.npy", idf.astype(np.float32))
        field_stats[f] = {"docs_nonempty": n_nonempty, "nnz": int(csc.nnz)}
        if verbose:
            print(f"  field '{f}': {n_nonempty:,} dokumen terisi, {csc.nnz:,} nilai non-nol", flush=True)
        del indices, counts, indptr, data, csr, csc

    ids_terms = [None] * n_terms
    for term, tid in vocab.items():
        ids_terms[tid] = term
    (out_dir / "vocab.json").write_text(json.dumps(ids_terms), encoding="utf-8")
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


#  3. CLI 
def main():
    ap = argparse.ArgumentParser(description="Bangun indeks TF-IDF pre-compute (VSM)")
    ap.add_argument("--input", default=str(DATASET_PATH), help="path dataset parquet")
    ap.add_argument("--output", default=str(INDEX_DIR), help="folder keluaran indeks")
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah dokumen (untuk uji cepat)")
    ap.add_argument("--batch-size", type=int, default=20_000)
    args = ap.parse_args()
    build_index_from_parquet(args.input, args.output, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()