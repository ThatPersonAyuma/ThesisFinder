import argparse
from pathlib import Path

import pandas as pd

from ..tfidf import STOPWORDS, VSM_DATA_DIR

INPUT = VSM_DATA_DIR / "frek_bersih.csv"
OUTPUT = VSM_DATA_DIR / "frek_stopword.csv"


def hapus_stopword(df: pd.DataFrame) -> pd.DataFrame:
    """Hilangkan stopword (daftar STOPWORDS di tfidf.py)."""
    if len(df) == 0:
        return df
    return df[~df["word"].isin(STOPWORDS)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="frek_bersih.csv -> frek_stopword.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"word": str}, keep_default_na=False)
    out = hapus_stopword(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"stopword dibuang: {len(df):,} -> {len(out):,} | {args.output}")


if __name__ == "__main__":
    main()
