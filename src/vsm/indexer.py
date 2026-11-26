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

from .cleaning.tokenizer import Tokenizer
from .tfidf import (
    DATASET_PATH,
    FIELD_ANALYZERS,
    FIELDS,
    INDEX_DIR,
    INDEX_FORMAT_VERSION,
    VOCAB_PATH,
    row_ids_from_indptr,
    weight_matrix_data,
)

READ_COLUMNS = ["id", "title", "abstract"]


def iter_papers(
    path,
    batch_size: int = 20_000,
    limit: int | None = None,
    fields: tuple[str, ...] = FIELDS,
) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(path)
    required_cols = ["id", *fields]
    missing = set(required_cols) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {sorted(missing)}")
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=required_cols):
        df = batch.to_pandas()
        if limit is not None and seen + len(df) > limit:
            df = df.iloc[: limit - seen]
        df["id"] = df["id"].astype(str)
        for f in fields:
            df[f] = df[f].fillna("").astype(str)
        yield df.reset_index(drop=True)
        seen += len(df)
        if limit is not None and seen >= limit:
            break


def _load_fixed_vocab(vocab) -> dict[str, int] | None:
    if vocab is None:
        return None
    if isinstance(vocab, Tokenizer):
        return dict(vocab.vocab)
    if isinstance(vocab, dict):
        return dict(vocab)
    if isinstance(vocab, (list, tuple)):
        return {t: i for i, t in enumerate(vocab)}
    tok = Tokenizer(vocab)
    return dict(tok.vocab)


def _prune_min_df(
    buf: dict,
    vocab: dict[str, int],
    fields: tuple[str, ...],
    min_df: int,
    df_global: list[int] | None = None,
):
    """Buang term dengan total kemunculan dokumen < min_df dan rapatkan ID term."""
    n_terms = len(vocab)
    if min_df <= 1 or n_terms == 0:
        return vocab
    df_total = np.zeros(n_terms, dtype=np.int64)
    for f in fields:
        idx_arr = np.frombuffer(buf[f]["indices"], dtype=np.int32)
        if len(idx_arr) > 0:
            df_total += np.bincount(idx_arr, minlength=n_terms)

    keep_mask = df_total >= min_df
    if keep_mask.all():
        return vocab

    remap = np.full(n_terms, -1, dtype=np.int32)
    kept_old_ids = np.flatnonzero(keep_mask)
    remap[kept_old_ids] = np.arange(len(kept_old_ids), dtype=np.int32)

    new_vocab = {term: int(remap[tid]) for term, tid in vocab.items() if remap[tid] >= 0}

    for f in fields:
        b = buf[f]
        old_idx = np.frombuffer(b["indices"], dtype=np.int32)
        old_cnt = np.frombuffer(b["counts"], dtype=np.int32)
        old_ptr = np.frombuffer(b["indptr"], dtype=np.int64)

        new_idx = array("i")
        new_cnt = array("i")
        new_ptr = array("q", [0])
        mapped = remap[old_idx] if len(old_idx) > 0 else np.empty(0, dtype=np.int32)
        for doc_i in range(len(old_ptr) - 1):
            start, end = int(old_ptr[doc_i]), int(old_ptr[doc_i + 1])
            for pos in range(start, end):
                nt = int(mapped[pos])
                if nt >= 0:
                    new_idx.append(nt)
                    new_cnt.append(int(old_cnt[pos]))
            new_ptr.append(len(new_idx))
        buf[f] = {"indices": new_idx, "counts": new_cnt, "indptr": new_ptr}

    if df_global is not None:
        df_global[:] = [df_global[int(i)] for i in kept_old_ids]

    return new_vocab


def build_index(
    batches,
    out_dir=INDEX_DIR,
    fields=FIELDS,
    vocab=None,
    min_df: int = 1,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(fields)
    for f in fields:
        if f not in FIELD_ANALYZERS:
            raise ValueError(f"Field tidak dikenal untuk indexing: {f}")

    t0 = time.time()
    fixed_vocab = _load_fixed_vocab(vocab)
    vocab_map: dict[str, int] = dict(fixed_vocab) if fixed_vocab is not None else {}

    buf = {f: {"indices": array("i"), "counts": array("i"), "indptr": array("q", [0])} for f in fields}
    meta_parts = []
    n_docs = 0
    # Statistik korpus untuk BM25F: df per dokumen (union semua field) dan total
    # token per field. `seen` mencegah term yang sama di title dan abstract
    # dihitung dua kali untuk dokumen yang sama. Untuk fixed vocab, statistik
    # diinisialisasi sepanjang vocab yang sudah ada.
    df_global: list[int] = [0] * len(vocab_map)
    seen: list[int] = [-1] * len(vocab_map)
    field_token_total = {f: 0 for f in fields}

    for batch in batches:
        if len(batch) == 0:
            continue
        base = n_docs
        columns = {f: batch[f].tolist() for f in fields}
        # Dokumen-major agar `seen` bisa mencegah term yang sama di title dan
        # abstract dihitung dua kali untuk satu dokumen.
        for doc_i in range(len(batch)):
            seq = base + doc_i
            for f in fields:
                b = buf[f]
                tokens = FIELD_ANALYZERS[f](columns[f][doc_i])
                for term, c in Counter(tokens).items():
                    if fixed_vocab is not None:
                        tid = vocab_map.get(term)
                        if tid is None:
                            continue
                    else:
                        tid = vocab_map.get(term)
                        if tid is None:
                            tid = vocab_map[term] = len(vocab_map)
                            df_global.append(0)
                            seen.append(-1)
                    if seen[tid] != seq:
                        df_global[tid] += 1
                        seen[tid] = seq
                    b["indices"].append(tid)
                    b["counts"].append(c)
                b["indptr"].append(len(b["indices"]))
                field_token_total[f] += len(tokens)
        meta_parts.append(batch[["id", "title"]])
        n_docs += len(batch)
        if verbose:
            print(f"  diproses {n_docs:,} dokumen | vocab {len(vocab_map):,} | {time.time() - t0:.0f}s", flush=True)

    if fixed_vocab is None and min_df > 1:
        vocab_map = _prune_min_df(buf, vocab_map, fields, min_df, df_global)

    n_terms = len(vocab_map)
    np.save(out_dir / "df_global.npy", np.asarray(df_global, dtype=np.int32))
    field_stats = {}
    for f in fields:
        b = buf[f]
        indices = np.frombuffer(b["indices"], dtype=np.int32)
        counts = np.frombuffer(b["counts"], dtype=np.int32)
        indptr = np.frombuffer(b["indptr"], dtype=np.int64)
        rows = row_ids_from_indptr(indptr)
        data, idf, n_nonempty = weight_matrix_data(indices, counts, indptr, n_terms, rows=rows)
        pd.DataFrame({
            "doc": rows.astype(np.int32),
            "term": indices,
            "w": data.astype(np.float32),
        }).to_parquet(out_dir / f"docvec_{f}.parquet", engine="pyarrow", index=False)
        np.save(out_dir / f"idf_{f}.npy", idf.astype(np.float32))
        df_arr = np.bincount(indices, minlength=n_terms) if len(indices) > 0 else np.zeros(n_terms, dtype=np.int32)
        np.save(out_dir / f"df_{f}.npy", df_arr.astype(np.int32))
        field_stats[f] = {"docs_nonempty": n_nonempty, "nnz": int(len(data))}
        if verbose:
            print(f"  field '{f}': {n_nonempty:,} dokumen terisi, {len(data):,} nilai non-nol", flush=True)
        del indices, counts, indptr, rows, data

    ids_terms = [""] * n_terms
    for term, tid in vocab_map.items():
        ids_terms[tid] = term
    (out_dir / "vocab.json").write_text(json.dumps(ids_terms), encoding="utf-8")

    meta_df = (
        pd.concat(meta_parts, ignore_index=True)
        if meta_parts
        else pd.DataFrame(columns=["id", "title"])
    )
    meta_df.to_parquet(out_dir / "meta.parquet", engine="pyarrow", index=False)

    config = {
        "format_version": INDEX_FORMAT_VERSION,
        "n_docs": n_docs,
        "n_terms": n_terms,
        "fields": list(fields),
        "vocab_source": str(vocab) if isinstance(vocab, (str, Path)) else ("custom" if fixed_vocab is not None else "corpus"),
        "min_df": min_df if fixed_vocab is None else None,
        "avgdl": {
            f: (field_token_total[f] / n_docs if n_docs else 0.0)
            for f in fields
        },
        "field_stats": field_stats,
        "weighting": "tf=1+ln(count); idf=ln((N+1)/(df+1))+1; L2-normalized rows; cosine",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if verbose:
        print(f"Selesai dalam {config['build_seconds']}s -> {out_dir}")
    return config


def build_index_from_parquet(
    path=DATASET_PATH,
    out_dir=INDEX_DIR,
    limit: int | None = None,
    batch_size: int = 20_000,
    fields=FIELDS,
    vocab=None,
    min_df: int = 1,
    verbose: bool = True,
) -> dict:
    return build_index(
        iter_papers(path, batch_size=batch_size, limit=limit, fields=tuple(fields)),
        out_dir=out_dir,
        fields=fields,
        vocab=vocab,
        min_df=min_df,
        verbose=verbose,
    )


def main():
    ap = argparse.ArgumentParser(description="Bangun indeks TF-IDF pre-compute (VSM: title + abstract)")
    ap.add_argument("--input", default=str(DATASET_PATH), help="path dataset parquet")
    ap.add_argument("--output", default=str(INDEX_DIR), help="folder keluaran indeks")
    ap.add_argument("--fields", nargs="+", default=list(FIELDS), help="field yang diindeks")
    ap.add_argument(
        "--vocab",
        default=str(VOCAB_PATH) if VOCAB_PATH.exists() else None,
        help="path vocab_clean.csv / vocab.json untuk membatasi kosakata bersih",
    )
    ap.add_argument("--no-vocab", action="store_true", help="bangun vocab langsung dari korpus tanpa vocab_clean.csv")
    ap.add_argument("--min-df", type=int, default=1, help="minimum df jika membangun vocab dari korpus")
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah dokumen (untuk uji cepat)")
    ap.add_argument("--batch-size", type=int, default=20_000)
    args = ap.parse_args()

    vocab_arg = None if args.no_vocab else args.vocab
    build_index_from_parquet(
        args.input,
        args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        fields=tuple(args.fields),
        vocab=vocab_arg,
        min_df=args.min_df,
    )


if __name__ == "__main__":
    main()
