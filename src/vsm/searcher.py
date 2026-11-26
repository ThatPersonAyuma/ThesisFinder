import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..algorithm.bm25f import BM25F
from .tfidf import (DATASET_PATH, DEFAULT_FIELD_WEIGHTS, FIELD_ANALYZERS, INDEX_DIR,
                    analyze_text, query_vector)

@dataclass
class SearchResult:
    rank: int
    doc_idx: int
    id: str
    title: str
    score: float
    field_scores: dict = field(default_factory=dict)


class VSMSearcher:
    def __init__(self, index_dir=INDEX_DIR, dataset_path=DATASET_PATH):
        index_dir = Path(index_dir)
        if not (index_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Indeks belum ada di {index_dir}. Jalankan: python -m src.vsm.indexer")
        self.config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        terms = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.n_docs = self.config["n_docs"]
        self.fields = tuple(self.config["fields"])
        self.terms = terms
        self.index_dir = index_dir
        # Vektor dokumen (CSR) dari disk diubah sekali ke CSC (kolom = term) sebagai inverted index
        # supaya query hanya menyentuh kolom term miliknya. Versi CSR-nya dibuang dari RAM.
        self.matrix = {f: self._load_docvec(f).tocsc() for f in self.fields}
        self.idf = {f: np.load(index_dir / f"idf_{f}.npy") for f in self.fields}
        self.meta = pd.read_parquet(index_dir / "meta.parquet")
        # Statistik korpus untuk BM25F; indeks lama mungkin belum punya.
        df_global_path = index_dir / "df_global.npy"
        self.df_global = np.load(df_global_path) if df_global_path.exists() else None
        if self.df_global is not None and len(self.df_global) != len(self.terms):
            self.df_global = None
        self.bm25_avgdl = self.config.get("avgdl") or None
        self.dataset_path = Path(dataset_path)
        self._dataset = None

    def _load_docvec(self, field: str) -> sp.csr_matrix:
        # docvec_<field>.parquet (doc, term, w) -> matriks CSR (baris = dokumen, kolom = term)
        d = pd.read_parquet(self.index_dir / f"docvec_{field}.parquet")
        doc, term, w = d["doc"].to_numpy(), d["term"].to_numpy(), d["w"].to_numpy()
        if len(doc) > 1 and (np.diff(doc) < 0).any():
            order = np.argsort(doc, kind="stable")
            doc, term, w = doc[order], term[order], w[order]
        indptr = np.concatenate(([0], np.cumsum(np.bincount(doc, minlength=self.n_docs)))).astype(np.int64)
        return sp.csr_matrix((w, term, indptr), shape=(self.n_docs, len(self.terms)))

    #  komponen internal 
    def _field_query(self, text: str, f: str):
        """Query string -> (term_ids, bobot) ternormalisasi untuk field f (term tak dikenal dibuang)."""
        counts = Counter()
        for tok in FIELD_ANALYZERS[f](text):
            tid = self.vocab.get(tok)
            if tid is not None:
                counts[tid] += 1
        return query_vector(counts, self.idf[f])

    def _resolve(self, query, fields, weights):
        """Susun {field: teks_query} dan {field: bobot} dari argumen pemanggil."""
        if isinstance(query, dict):
            qmap = {f: q for f, q in query.items() if q and q.strip()}
            unknown = set(qmap) - set(self.fields)
            if unknown:
                raise ValueError(f"Field tidak dikenal: {sorted(unknown)}; pilihan: {list(self.fields)}")
        else:
            chosen = list(fields) if fields else list(self.fields)
            unknown = set(chosen) - set(self.fields)
            if unknown:
                raise ValueError(f"Field tidak dikenal: {sorted(unknown)}; pilihan: {list(self.fields)}")
            qmap = {f: query for f in chosen}
        w = {f: float((weights or {}).get(f, DEFAULT_FIELD_WEIGHTS.get(f, 1.0))) for f in qmap}
        total = sum(w.values())
        if total <= 0:
            raise ValueError("Jumlah bobot field harus > 0")
        return qmap, {f: v / total for f, v in w.items()}

    def _candidate_mask(self, candidates):
        if candidates is None:
            return None
        arr = np.asarray(candidates)
        if arr.dtype == bool:
            if arr.shape != (self.n_docs,):
                raise ValueError("Mask kandidat harus sepanjang jumlah dokumen indeks")
            return arr
        mask = np.zeros(self.n_docs, dtype=bool)
        mask[arr.astype(np.int64)] = True
        return mask

    #  API publik 
    def score_all(self, query, fields=None, weights=None, candidates=None):
        """Skor cosine terbobot untuk SEMUA dokumen. Mengembalikan (skor_total, {field: skor_field})."""
        qmap, w = self._resolve(query, fields, weights)
        total = np.zeros(self.n_docs, dtype=np.float32)
        per_field = {}
        for f, text in qmap.items():
            ids, qw = self._field_query(text, f)
            if len(ids) == 0:
                per_field[f] = np.zeros(self.n_docs, dtype=np.float32)
                continue
            s = self.matrix[f][:, ids].dot(qw.astype(np.float32))
            per_field[f] = np.asarray(s, dtype=np.float32).ravel()
            total += w[f] * per_field[f]
        mask = self._candidate_mask(candidates)
        if mask is not None:
            total[~mask] = 0.0
        return total, per_field

    def doc_vector(self, doc_idx: int, field: str, top: int | None = None) -> dict:
        # Vektor TF-IDF tersimpan milik satu dokumen pada satu field: {term: bobot}, bobot terbesar dulu.
        # Dibaca dari docvec_<field>.parquet (field kosong -> {}).
        if field not in self.fields:
            raise ValueError(f"Field tidak dikenal: {field}; pilihan: {list(self.fields)}")
        if not hasattr(self, "_docvec_csr"):
            self._docvec_csr = {}
        if field not in self._docvec_csr:
            self._docvec_csr[field] = self._load_docvec(field)
        row = self._docvec_csr[field][int(doc_idx)]
        order = np.argsort(-row.data)
        if top is not None:
            order = order[:top]
        return {self.terms[int(row.indices[j])]: float(row.data[j]) for j in order}

    def top_candidates(self, query, fields=None, weights=None, top_n=1000, candidates=None):
        """Top-N dokumen (doc_idx, skor) urut skor menurun; dokumen berskor 0 tidak disertakan."""
        total, _ = self.score_all(query, fields, weights, candidates)
        return self._top(total, top_n)

    def search(self, query, fields=None, weights=None, top_k=10, candidates=None):
        """Pencarian lengkap -> list[SearchResult].

        query  : str (dicari di semua `fields` terpilih) ATAU dict {field: teks} untuk
                 pencarian multi-field dengan query berbeda per field,
                 mis. {"title": "graph neural network", "abstract": "molecule"}.
        fields : daftar field yang dipakai bila query berupa str (default: semua field).
        weights: dict bobot per field (opsional; default di config.DEFAULT_FIELD_WEIGHTS).
        """
        total, per_field = self.score_all(query, fields, weights, candidates)
        idx, scores = self._top(total, top_k)
        out = []
        for rank, (i, s) in enumerate(zip(idx, scores), start=1):
            row = self.meta.iloc[int(i)]
            out.append(SearchResult(
                rank=rank, doc_idx=int(i), id=row["id"], title=row["title"], score=float(s),
                field_scores={f: float(per_field[f][i]) for f in per_field}))
        return out

    def _load_dataset(self) -> pd.DataFrame:
        # BM25 butuh teks mentah, bukan vektor TF-IDF; cache agar tidak dibaca tiap query.
        if self._dataset is None:
            self._dataset = pd.read_parquet(self.dataset_path, columns=["title", "abstract"])
        return self._dataset

    def _corpus_idf(self, text):
        """IDF BM25F dari df seluruh korpus; None jika indeks lama belum punya df_global."""
        if self.df_global is None:
            return None
        idf = {}
        for term in set(analyze_text(text)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            df = min(max(int(self.df_global[tid]), 0), self.n_docs)
            idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
        return idf

    def search_bm25(self, query, top_k=10, candidate_k=100, candidates=None):
        """VSM mengambil `candidate_k` kandidat, lalu BM25F mengurutkan ulang.

        Query dict {field: teks} digabung karena BM25F memakai satu query untuk
        semua field. IDF dan avgdl memakai statistik korpus dari indeks; jika
        indeks lama belum menyimpannya, fallback ke statistik kandidat. Hasil
        memakai doc_idx yang sama dengan indeks VSM.
        """
        text = " ".join(str(q) for q in query.values() if q) if isinstance(query, dict) else query
        if not text or not str(text).strip() or (top_k is not None and top_k <= 0):
            return []
        if candidate_k is not None and top_k is not None:
            candidate_k = max(candidate_k, top_k)

        cand_idx, _ = self.top_candidates(text, top_n=candidate_k, candidates=candidates)
        if len(cand_idx) == 0:
            return []

        dataset = self._load_dataset()
        docs = []
        for i in cand_idx:
            row = dataset.iloc[int(i)]
            docs.append({
                "title": "" if pd.isna(row["title"]) else str(row["title"]),
                "abstract": "" if pd.isna(row["abstract"]) else str(row["abstract"]),
            })

        model = BM25F(docs, tokenizer=analyze_text,
                      corpus_idf=self._corpus_idf(text),
                      field_avgdl=self.bm25_avgdl)
        out = []
        for rank, (local_id, score) in enumerate(model.search(text, top_k=top_k), start=1):
            i = int(cand_idx[local_id])
            row = self.meta.iloc[i]
            out.append(SearchResult(
                rank=rank, doc_idx=i, id=row["id"], title=row["title"], score=float(score)))
        return out

    @staticmethod
    def _top(scores: np.ndarray, k):
        nz = np.flatnonzero(scores > 0)
        if len(nz) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if k is not None and len(nz) > k:
            nz = nz[np.argpartition(-scores[nz], k - 1)[:k]]
        order = np.lexsort((nz, -scores[nz]))  # skor menurun, tie-break indeks naik
        nz = nz[order]
        return nz, scores[nz]


# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(description="Cari artikel dengan VSM (TF-IDF) atau BM25F")
    ap.add_argument("query", nargs="?", default=None, help="query untuk semua field terpilih")
    ap.add_argument("--fields", nargs="*", default=None, help="field yang dipakai (title abstract)")
    for f in ("title", "abstract"):
        ap.add_argument(f"--{f}", default=None, help=f"query khusus field {f}")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--bm25", action="store_true", help="rerank kandidat VSM dengan BM25F")
    ap.add_argument("--candidate-k", type=int, default=100,
                    help="jumlah kandidat VSM yang di-rerank BM25 (default: 100)")
    ap.add_argument("--index", default=None, help="folder indeks (default: data/index/vsm)")
    args = ap.parse_args()

    per_field = {f: getattr(args, f) for f in ("title", "abstract") if getattr(args, f)}
    query = per_field if per_field else args.query
    if not query:
        ap.error("isi query atau salah satu --title/--abstract")

    s = VSMSearcher(args.index) if args.index else VSMSearcher()
    t = time.perf_counter()
    if args.bm25:
        results = s.search_bm25(query, top_k=args.top, candidate_k=args.candidate_k)
    else:
        results = s.search(query, fields=args.fields, top_k=args.top)
    ms = (time.perf_counter() - t) * 1000
    print(f"{len(results)} hasil ({ms:.1f} ms)\n")
    for r in results:
        fs = " ".join(f"{k[:3]}={v:.2f}" for k, v in r.field_scores.items()) or "bm25"
        print(f"{r.rank:>2}. [{r.score:.3f}] {r.title}\n    {r.id} | {fs}")


if __name__ == "__main__":
    main()