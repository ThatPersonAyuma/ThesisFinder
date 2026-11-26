"""
Search Pipeline - Orchestration lengkap Binary Filtering, VSM Retrieval, dan BM25F Re-ranking.

Workflow:
1. Pre-filtering (Binary Model): year + primary_category -> candidates
2. Stage-1 Retrieval (VSM TF-IDF): cosine similarity cepat atas candidates
3. Stage-2 Re-ranking (BM25F): peringkat multi-field dengan tokenizer & IDF korpus sinkron
4. Post-filtering (Binary Model): authors + categories
5. Metadata enrichment
"""
import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.algorithm.bm25f import BM25F
from src.binary import BinaryFilter
from src.binary.config import BINARY_INDEX_DIR, DATASET_PATH
from src.pipeline.setup import ensure_indexes
from src.vsm.searcher import VSMSearcher
from src.vsm.tfidf import INDEX_DIR as VSM_INDEX_DIR
from src.vsm.tfidf import analyze_text

STRUCTURE_COLUMNS = ("keywords", "metode", "evaluasi", "kesimpulan")


class SearchPipeline:
    """Pipeline pencarian lengkap: Binary Filter + VSM TF-IDF + BM25F."""

    def __init__(
        self,
        dataset_path: str | Path = DATASET_PATH,
        vsm_index_dir: str | Path = VSM_INDEX_DIR,
        binary_index_dir: str | Path = BINARY_INDEX_DIR,
        structure_path: str | Path | None = None,
        verbose: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.vsm = VSMSearcher(vsm_index_dir)
        self.bf = BinaryFilter(binary_index_dir, dataset_path=self.dataset_path)
        self._dataset: pd.DataFrame | None = None
        self._structure_by_id: dict[str, dict[str, str]] | None = None
        self.structure_path = Path(structure_path) if structure_path else None

        # Muat df korpus per field dari indeks VSM untuk estimasi IDF BM25F tingkat korpus.
        self._df_arrays = {}
        for f in self.vsm.fields:
            df_path = self.vsm.index_dir / f"df_{f}.npy"
            if df_path.exists():
                self._df_arrays[f] = np.load(df_path)

        if verbose:
            print("SearchPipeline initialized:")
            print(f"  - VSM index: {self.vsm.n_docs:,} documents ({len(self.vsm.terms):,} terms)")
            print(f"  - Binary filter: {self.bf}")
            print(f"  - Dataset: {self.dataset_path}")

    @property
    def dataset(self) -> pd.DataFrame:
        """Lazy load dataset agar inisialisasi cepat dan hemat RAM."""
        if self._dataset is None:
            self._dataset = pd.read_parquet(self.dataset_path)
        return self._dataset

    def _get_structure_map(self) -> dict[str, dict[str, str]]:
        if self._structure_by_id is not None:
            return self._structure_by_id
        self._structure_by_id = {}
        if self.structure_path is not None and self.structure_path.exists():
            sdf = pd.read_parquet(self.structure_path)
            avail = [c for c in STRUCTURE_COLUMNS if c in sdf.columns]
            if "id" in sdf.columns and avail:
                for row in sdf[["id", *avail]].itertuples(index=False):
                    pid = str(row[0])
                    self._structure_by_id[pid] = {
                        col: "" if pd.isna(val) else str(val)
                        for col, val in zip(avail, row[1:])
                    }
        return self._structure_by_id

    def _corpus_bm25f_idf(self, query: str) -> dict[str, float]:
        """Hitung IDF BM25F tingkat korpus untuk term query dari df VSM."""
        if not self._df_arrays or self.vsm.n_docs == 0:
            return {}
        n_docs = float(self.vsm.n_docs)
        idf_map: dict[str, float] = {}
        for term in set(analyze_text(query)):
            tid = self.vsm.vocab.get(term)
            if tid is None:
                continue
            # Gabungkan probabilitas kemunculan antar-field (title / abstract) agar df <= N.
            p_not_seen = 1.0
            for arr in self._df_arrays.values():
                df_f = min(float(arr[tid]), n_docs)
                p_not_seen *= 1.0 - (df_f / n_docs)
            df_est = n_docs * (1.0 - p_not_seen)
            idf_map[term] = math.log((n_docs - df_est + 0.5) / (df_est + 0.5) + 1.0)
        return idf_map

    def _build_doc_dict(self, row: pd.Series, extra: dict[str, str] | None = None) -> dict[str, str]:
        doc = {
            "title": "" if pd.isna(row["title"]) else str(row["title"]),
            "abstract": "" if pd.isna(row["abstract"]) else str(row["abstract"]),
        }
        if "keywords" in row.index and pd.notna(row["keywords"]):
            kw = row["keywords"]
            if isinstance(kw, (list, np.ndarray)):
                doc["keywords"] = " ".join(str(x) for x in kw)
            else:
                doc["keywords"] = str(kw)
        if extra:
            for k, v in extra.items():
                if v:
                    doc[k] = v
        return doc

    def search(
        self,
        query: str,
        year_range: Optional[Tuple[float, float]] = None,
        primary_categories: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        all_categories: Optional[List[str]] = None,
        top_k: int = 10,
        mode: str = "hybrid",
        rerank_k: Optional[int] = None,
        bm25f_weights: Optional[Dict[str, float]] = None,
        vsm_weights: Optional[Dict[str, float]] = None,
        hybrid_alpha: float = 0.7,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Pencarian terintegrasi Binary + VSM + BM25F.

        Args:
            mode: "hybrid" (VSM stage-1 -> BM25F stage-2), "vsm", atau "bm25f".
            hybrid_alpha: Bobot skor BM25F ternormalisasi dalam mode hybrid (0..1; sisanya VSM).
        """
        if mode not in ("hybrid", "vsm", "bm25f"):
            raise ValueError(f"Mode tidak dikenal: {mode!r}. Pilihan: 'hybrid', 'vsm', 'bm25f'")
        if not query or not str(query).strip() or top_k <= 0:
            return []

        # Step 1: Pre-filtering (Binary Model)
        candidates = None
        if year_range is not None or primary_categories is not None:
            all_candidates = self.bf.get_pre_filter_candidates(
                year_range=year_range,
                primary_categories=primary_categories,
            )
            max_docs = min(self.vsm.n_docs, len(self.dataset)) if mode != "bm25f" else len(self.dataset)
            candidates = all_candidates[all_candidates < max_docs]
            if len(candidates) == 0:
                if verbose:
                    print(f"Pre-filtering: {len(all_candidates):,} total, 0 di indeks aktif")
                return []
            if verbose:
                print(f"Pre-filtering: {len(candidates):,} kandidat")

        has_post_filters = authors is not None or all_categories is not None
        pool_k = rerank_k
        if pool_k is None:
            if has_post_filters:
                pool_k = max(top_k * 30, 500)
            elif mode == "hybrid":
                pool_k = max(top_k * 10, 100)
            else:
                pool_k = top_k

        # Step 2 & 3: Retrieval & Re-ranking
        ranked_entries: list[tuple[int, float, float, float | None, dict]] = []
        # tuple: (doc_idx, final_score, vsm_score, bm25f_score, field_scores)

        if mode == "bm25f" and (
            (candidates is not None and len(candidates) <= 5_000)
            or (candidates is None and len(self.dataset) <= 5_000)
        ):
            # Jalankan BM25F langsung atas kandidat / korpus kecil.
            target_ids = (
                candidates.tolist() if candidates is not None else list(range(len(self.dataset)))
            )
            struct_map = self._get_structure_map()
            docs_by_id = []
            for doc_idx in target_ids:
                row = self.dataset.iloc[int(doc_idx)]
                docs_by_id.append(
                    (int(doc_idx), self._build_doc_dict(row, struct_map.get(str(row["id"]))))
                )
            corpus_idf = self._corpus_bm25f_idf(query) if len(self.dataset) > len(target_ids) else None
            bm25_ranked = BM25F.rerank(
                query,
                docs_by_id,
                field_weights=bm25f_weights,
                tokenizer=analyze_text,
                corpus_idf=corpus_idf,
                fallback_missing_idf=True,
                top_k=pool_k if has_post_filters else top_k,
            )
            for doc_idx, bm_score in bm25_ranked:
                ranked_entries.append((doc_idx, bm_score, 0.0, bm_score, {}))
        else:
            # Stage 1: VSM TF-IDF
            vsm_results = self.vsm.search(
                query=query,
                weights=vsm_weights,
                candidates=candidates,
                top_k=pool_k,
            )
            if not vsm_results:
                return []
            if verbose:
                print(f"VSM retrieval: {len(vsm_results)} kandidat")

            if mode == "vsm":
                for r in vsm_results:
                    ranked_entries.append((r.doc_idx, r.score, r.score, None, r.field_scores))
            else:
                # Stage 2: BM25F Re-ranking (mode == "hybrid" atau fallback korpus besar mode == "bm25f")
                struct_map = self._get_structure_map()
                docs_by_id = []
                for r in vsm_results:
                    row = self.dataset.iloc[r.doc_idx]
                    docs_by_id.append(
                        (r.doc_idx, self._build_doc_dict(row, struct_map.get(str(row["id"]))))
                    )
                corpus_idf = self._corpus_bm25f_idf(query)
                bm25_ranked = BM25F.rerank(
                    query,
                    docs_by_id,
                    field_weights=bm25f_weights,
                    tokenizer=analyze_text,
                    corpus_idf=corpus_idf,
                    fallback_missing_idf=True,
                )
                bm_scores = {doc_idx: score for doc_idx, score in bm25_ranked}
                max_bm = max(bm_scores.values(), default=0.0)

                for r in vsm_results:
                    bm_raw = bm_scores.get(r.doc_idx, 0.0)
                    if mode == "bm25f":
                        final = bm_raw
                    else:
                        bm_norm = (bm_raw / max_bm) if max_bm > 0 else 0.0
                        final = hybrid_alpha * bm_norm + (1.0 - hybrid_alpha) * r.score
                    if final > 0:
                        ranked_entries.append((r.doc_idx, float(final), float(r.score), float(bm_raw), r.field_scores))

                ranked_entries.sort(key=lambda item: (-item[1], item[0]))

        if not ranked_entries:
            return []

        # Step 4: Post-filtering (Binary Model) dengan mempertahankan urutan skor
        if has_post_filters:
            doc_ids = np.array([item[0] for item in ranked_entries], dtype=np.int64)
            filtered_doc_ids = self.bf.apply_post_filters(
                doc_ids=doc_ids,
                authors=authors,
                categories=all_categories,
            )
            if len(filtered_doc_ids) == 0:
                return []
            allowed = set(int(x) for x in filtered_doc_ids)
            ranked_entries = [item for item in ranked_entries if item[0] in allowed]
            if verbose:
                print(f"Post-filtering: {len(ranked_entries)} dokumen lolos")

        ranked_entries = ranked_entries[:top_k]

        # Step 5: Metadata enrichment
        results = []
        for rank, (doc_idx, final_score, vsm_score, bm25f_score, field_scores) in enumerate(
            ranked_entries, start=1
        ):
            doc_row = self.dataset.iloc[doc_idx]
            authors_val = doc_row["authors"]
            cats_val = doc_row["categories"]
            kw_val = doc_row["keywords"] if "keywords" in doc_row.index else None

            enriched = {
                "rank": rank,
                "doc_idx": doc_idx,
                "score": float(final_score),
                "vsm_score": float(vsm_score),
                "bm25f_score": None if bm25f_score is None else float(bm25f_score),
                "field_scores": field_scores,
                "id": doc_row["id"],
                "title": doc_row["title"],
                "abstract": doc_row["abstract"] if pd.notna(doc_row["abstract"]) else None,
                "authors": list(authors_val) if isinstance(authors_val, (np.ndarray, list, tuple)) else [],
                "categories": list(cats_val) if isinstance(cats_val, (np.ndarray, list, tuple)) else [],
                "primary_category": doc_row["primary_category"] if pd.notna(doc_row["primary_category"]) else None,
                "year": int(doc_row["year"]) if pd.notna(doc_row["year"]) else None,
                "keywords": kw_val if isinstance(kw_val, str) and kw_val else None,
            }
            results.append(enriched)

        return results

    def rerank_ids(
        self,
        query: str,
        doc_ids: str | list[str],
        top_k: int | None = None,
        bm25f_weights: dict | None = None,
    ) -> list[dict]:
        """Re-rank dokumen tertentu berdasarkan ``id`` dataset dengan BM25F.

        ``doc_ids`` boleh satu id atau list id (mis. URL arXiv). Hanya dokumen
        yang ditemukan di dataset yang dipertimbangkan.
        """
        if not query or not str(query).strip():
            return []
        if top_k is not None and top_k <= 0:
            return []
        if doc_ids is None:
            return []
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        wanted = {str(x) for x in doc_ids}
        if not wanted:
            return []

        mask = self.dataset["id"].astype(str).isin(wanted).to_numpy()
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            return []

        struct_map = self._get_structure_map()
        docs_by_id = []
        for pos in positions:
            pos = int(pos)
            row = self.dataset.iloc[pos]
            docs_by_id.append(
                (pos, self._build_doc_dict(row, struct_map.get(str(row["id"]))))
            )

        bm25_ranked = BM25F.rerank(
            query,
            docs_by_id,
            field_weights=bm25f_weights,
            tokenizer=analyze_text,
            corpus_idf=self._corpus_bm25f_idf(query),
            fallback_missing_idf=True,
            top_k=top_k,
        )
        results = []
        for rank, (doc_idx, score) in enumerate(bm25_ranked, start=1):
            row = self.dataset.iloc[doc_idx]
            results.append({
                "rank": rank,
                "doc_idx": doc_idx,
                "id": row["id"],
                "title": row["title"],
                "score": float(score),
                "bm25f_score": float(score),
            })
        return results

    def __repr__(self):
        return f"SearchPipeline(vsm={self.vsm.n_docs} docs, bf={self.bf})"


# ===== CLI =====
def main():
    parser = argparse.ArgumentParser(
        description="ThesisFinder Search - Pencarian terintegrasi Binary + VSM + BM25F"
    )
    parser.add_argument("query", help="Query string untuk pencarian")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vsm", "bm25f"],
        default="hybrid",
        help="Mode retrieval (default: hybrid = VSM + BM25F rerank)",
    )
    parser.add_argument("--rerank-k", type=int, default=None, help="Jumlah kandidat VSM untuk BM25F")
    parser.add_argument("--year-min", type=float, default=None, help="Tahun minimum (inclusive)")
    parser.add_argument("--year-max", type=float, default=None, help="Tahun maksimum (inclusive)")
    parser.add_argument("--primary-categories", nargs="+", default=None, help="Primary categories (OR)")
    parser.add_argument("--authors", nargs="+", default=None, help="Author names untuk post-filtering")
    parser.add_argument("--all-categories", nargs="+", default=None, help="Categories untuk post-filtering")
    parser.add_argument("--top", type=int, default=10, help="Jumlah maksimal hasil (default: 10)")
    parser.add_argument(
        "--no-auto-build",
        action="store_true",
        help="Jangan bangun indeks otomatis bila belum tersedia",
    )

    args = parser.parse_args()

    year_range = None
    if args.year_min is not None or args.year_max is not None:
        year_min = args.year_min if args.year_min is not None else 0
        year_max = args.year_max if args.year_max is not None else 9999
        year_range = (year_min, year_max)

    if not args.no_auto_build:
        ensure_indexes(verbose=True)
    print("Initializing pipeline...\n")
    pipeline = SearchPipeline()

    print(f"\nSearching ({args.mode}) for: '{args.query}'")
    if year_range:
        print(f"  Year range: {year_range[0]} - {year_range[1]}")
    if args.primary_categories:
        print(f"  Primary categories: {args.primary_categories}")
    if args.authors:
        print(f"  Authors: {args.authors}")
    if args.all_categories:
        print(f"  Categories: {args.all_categories}")
    print()

    start_time = time.perf_counter()
    results = pipeline.search(
        query=args.query,
        year_range=year_range,
        primary_categories=args.primary_categories,
        authors=args.authors,
        all_categories=args.all_categories,
        top_k=args.top,
        mode=args.mode,
        rerank_k=args.rerank_k,
        verbose=True,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n{'='*80}")
    print(f"Found {len(results)} results in {elapsed_ms:.1f} ms")
    print(f"{'='*80}\n")

    for result in results:
        bm_str = f" | bm25f={result['bm25f_score']:.3f}" if result["bm25f_score"] is not None else ""
        print(
            f"{result['rank']:>2}. [Score: {result['score']:.4f} | vsm={result['vsm_score']:.3f}{bm_str}] "
            f"{result['title']}"
        )
        print(f"    ID: {result['id']}")
        authors_str = ", ".join(result["authors"]) if result["authors"] else "N/A"
        if len(authors_str) > 80:
            authors_str = authors_str[:77] + "..."
        print(f"    Authors: {authors_str}")
        year_str = str(result["year"]) if result["year"] else "N/A"
        primary_cat_str = result["primary_category"] if result["primary_category"] else "N/A"
        print(f"    Year: {year_str} | Primary Category: {primary_cat_str}")
        categories_str = ", ".join(result["categories"]) if result["categories"] else "N/A"
        if len(categories_str) > 80:
            categories_str = categories_str[:77] + "..."
        print(f"    Categories: {categories_str}")
        if result["abstract"]:
            abstract_preview = result["abstract"][:150].replace("\n", " ")
            if len(result["abstract"]) > 150:
                abstract_preview += "..."
            print(f"    Abstract: {abstract_preview}")
        print()


if __name__ == "__main__":
    main()
