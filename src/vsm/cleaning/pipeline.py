"""Pipeline pembersihan data & pembentukan kosakata VSM dalam satu langkah."""

import argparse
from pathlib import Path

from ..tfidf import DATASET_PATH, VSM_DATA_DIR
from .cleaning import MIN_FREQ, build_word_freq_from_parquet, clean_frequencies, read_freq_csv
from .stemming import stemming
from .stopword import hapus_stopword
from .tokenizer import build_vocab_csv


def run_cleaning_pipeline(
    input_csv: str | Path | None = None,
    data_dir: str | Path = VSM_DATA_DIR,
    min_freq: int = MIN_FREQ,
    from_parquet: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Jalankan cleaning -> stopword -> stemming -> vocab_clean.csv secara terpadu."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(input_csv) if input_csv else data_dir / "frek.csv"
    clean_path = data_dir / "frek_bersih.csv"
    stop_path = data_dir / "frek_stopword.csv"
    stem_path = data_dir / "frek_stem.csv"
    vocab_path = data_dir / "vocab_clean.csv"

    if from_parquet is not None:
        if verbose:
            print(f"[0/4] Ekstrak frekuensi kata dari {from_parquet}...")
        df_raw = build_word_freq_from_parquet(from_parquet, raw_path)
    else:
        df_raw = read_freq_csv(raw_path)

    if verbose:
        print(f"[1/4] Cleaning {raw_path} ({len(df_raw):,} baris)...")
    df_clean = clean_frequencies(df_raw, min_freq=min_freq, verbose=verbose)
    df_clean.to_csv(clean_path, index=False)

    if verbose:
        print(f"[2/4] Stopword removal -> {stop_path}...")
    df_stop = hapus_stopword(df_clean)
    df_stop.to_csv(stop_path, index=False)
    if verbose:
        print(f"  stopword dibuang: {len(df_clean):,} -> {len(df_stop):,}")

    if verbose:
        print(f"[3/4] Stemming ringan -> {stem_path}...")
    df_stem = stemming(df_stop)
    df_stem.to_csv(stem_path, index=False)
    if verbose:
        print(f"  stemming: {len(df_stop):,} -> {len(df_stem):,}")

    if verbose:
        print(f"[4/4] Membangun vocab_clean -> {vocab_path}...")
    df_vocab = build_vocab_csv(df_stem)
    df_vocab.to_csv(vocab_path, index=False)
    if verbose:
        print(f"Selesai: {len(df_vocab):,} term tersimpan di {vocab_path}")

    return vocab_path


def main():
    ap = argparse.ArgumentParser(description="Pipeline lengkap pembersihan vocab VSM")
    ap.add_argument("--input", default=None, help="path frek.csv (default: data/vsm/frek.csv)")
    ap.add_argument("--data-dir", default=str(VSM_DATA_DIR), help="folder output CSV VSM")
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument(
        "--from-parquet",
        nargs="?",
        const=str(DATASET_PATH),
        default=None,
        help="bangun ulang frek.csv langsung dari dataset parquet",
    )
    args = ap.parse_args()
    run_cleaning_pipeline(
        input_csv=args.input,
        data_dir=args.data_dir,
        min_freq=args.min_freq,
        from_parquet=args.from_parquet,
        verbose=True,
    )


if __name__ == "__main__":
    main()
