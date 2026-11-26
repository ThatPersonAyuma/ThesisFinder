"""Utilitas setup agar indeks VSM dan Binary siap pakai setelah clone."""

import argparse
import json
from pathlib import Path

from src.binary.config import BINARY_INDEX_DIR, DATASET_PATH
from src.binary.indexer import BinaryIndexer
from src.vsm.indexer import build_index_from_parquet
from src.vsm.tfidf import INDEX_DIR as VSM_INDEX_DIR
from src.vsm.tfidf import VOCAB_PATH


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size > 1024:
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def ensure_indexes(
    dataset_path: str | Path = DATASET_PATH,
    vsm_index_dir: str | Path = VSM_INDEX_DIR,
    binary_index_dir: str | Path = BINARY_INDEX_DIR,
    vocab_path: str | Path | None = VOCAB_PATH,
    verbose: bool = True,
) -> dict:
    """Pastikan indeks VSM dan Binary tersedia dan sinkron; bangun otomatis jika belum ada."""
    dataset_path = Path(dataset_path)
    vsm_index_dir = Path(vsm_index_dir)
    binary_index_dir = Path(binary_index_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {dataset_path}. Pastikan git LFS sudah terunduh (git lfs pull)."
        )
    if _is_lfs_pointer(dataset_path):
        raise RuntimeError(
            f"File {dataset_path} masih berupa pointer Git LFS. Jalankan 'git lfs pull' terlebih dahulu."
        )

    vsm_cfg_path = vsm_index_dir / "config.json"
    bin_cfg_path = binary_index_dir / "config.json"

    built_vsm = False
    built_binary = False

    if not vsm_cfg_path.exists():
        if verbose:
            print(f"[Setup] Indeks VSM belum ada di {vsm_index_dir}. Membangun indeks VSM...")
        use_vocab = None
        min_df = 5
        if vocab_path is not None:
            vp = Path(vocab_path)
            if vp.exists() and not _is_lfs_pointer(vp):
                use_vocab = vp
                min_df = 1
        build_index_from_parquet(
            dataset_path,
            vsm_index_dir,
            vocab=use_vocab,
            min_df=min_df,
            verbose=verbose,
        )
        built_vsm = True

    if not bin_cfg_path.exists():
        if verbose:
            print(f"[Setup] Indeks Binary belum ada di {binary_index_dir}. Membangun indeks Binary...")
        BinaryIndexer(dataset_path=dataset_path, output_dir=binary_index_dir).build_indexes(
            verbose=verbose
        )
        built_binary = True

    vsm_cfg = json.loads(vsm_cfg_path.read_text(encoding="utf-8"))
    bin_cfg = json.loads(bin_cfg_path.read_text(encoding="utf-8"))

    if vsm_cfg.get("n_docs") != bin_cfg.get("total_docs"):
        if verbose:
            print(
                f"[Setup] Jumlah dokumen tidak sinkron (VSM={vsm_cfg.get('n_docs')}, "
                f"Binary={bin_cfg.get('total_docs')}). Membangun ulang indeks Binary..."
            )
        BinaryIndexer(dataset_path=dataset_path, output_dir=binary_index_dir).build_indexes(
            verbose=verbose
        )
        bin_cfg = json.loads(bin_cfg_path.read_text(encoding="utf-8"))
        built_binary = True

    return {
        "built_vsm": built_vsm,
        "built_binary": built_binary,
        "vsm_docs": vsm_cfg.get("n_docs"),
        "binary_docs": bin_cfg.get("total_docs"),
    }


def main():
    ap = argparse.ArgumentParser(description="Bangun indeks VSM dan Binary jika belum tersedia")
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--vsm-index", default=str(VSM_INDEX_DIR))
    ap.add_argument("--binary-index", default=str(BINARY_INDEX_DIR))
    args = ap.parse_args()

    status = ensure_indexes(
        dataset_path=args.dataset,
        vsm_index_dir=args.vsm_index,
        binary_index_dir=args.binary_index,
        verbose=True,
    )
    print(f"Indeks siap: VSM={status['vsm_docs']:,} docs | Binary={status['binary_docs']:,} docs")


if __name__ == "__main__":
    main()
