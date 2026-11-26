import argparse

import pandas as pd

from ..tfidf import ROOT_DIR, STOPWORDS

INPUT = ROOT_DIR / "frek_bersih.csv"
OUTPUT = ROOT_DIR / "frek_stopword.csv"


# Menghilangkan stopword (daftar STOPWORDS di tfidf.py)
def hapus_stopword(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["word"].isin(STOPWORDS)]


def main():
    ap = argparse.ArgumentParser(description="frek_bersih.csv -> frek_stopword.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"word": str}, keep_default_na=False)
    out = hapus_stopword(df)
    out.to_csv(args.output, index=False)
    print(f"stopword dibuang: {len(df):,} -> {len(out):,} | {args.output}")


if __name__ == "__main__":
    main()