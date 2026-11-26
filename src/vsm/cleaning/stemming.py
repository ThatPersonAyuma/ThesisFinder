import argparse
from pathlib import Path

import pandas as pd

from ..tfidf import VSM_DATA_DIR, light_stem

INPUT = VSM_DATA_DIR / "frek_stopword.csv"
OUTPUT = VSM_DATA_DIR / "frek_stem.csv"


def stemming(df: pd.DataFrame) -> pd.DataFrame:
    """Stemming ringan (light_stem di tfidf.py), lalu gabungkan frekuensi term yang sama."""
    if len(df) == 0:
        return df
    out = df.assign(word=df["word"].map(light_stem)).groupby("word", as_index=False)["frequence"].sum()
    return out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="frek_stopword.csv -> frek_stem.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"word": str}, keep_default_na=False)
    out = stemming(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"stemming: {len(df):,} -> {len(out):,} | {args.output}")


if __name__ == "__main__":
    main()
