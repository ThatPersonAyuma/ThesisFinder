import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from ..tfidf import ROOT_DIR, analyze_text

INPUT = ROOT_DIR / "frek_stem.csv"
CLEAN_PATH = ROOT_DIR / "vocab_clean.csv"


class Tokenizer:
    """Teks -> token: lowercase, stopword, stemming (analyze_text), lalu hanya term yang ada di vocab_clean.csv."""

    def __init__(self, vocab_path=CLEAN_PATH):
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocab belum ada di {path}. Jalankan: python -m src.vsm.cleaning.tokenizer")
        self.source = path
        self.terms = pd.read_csv(path, dtype={"word": str}, keep_default_na=False)["word"].tolist()
        self.vocab = {t: i for i, t in enumerate(self.terms)}

    def __len__(self) -> int:
        return len(self.terms)

    def tokenize(self, text: str) -> list[str]:
        return [t for t in analyze_text(text) if t in self.vocab]

    def encode(self, text: str) -> Counter:
        return Counter(self.vocab[t] for t in self.tokenize(text))


# Membuat vocab_clean.csv (id, word, frequence) dari frek_stem.csv; id = nomor baris = id term di indeks
def main():
    ap = argparse.ArgumentParser(description="frek_stem.csv -> vocab_clean.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(CLEAN_PATH))
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"word": str}, keep_default_na=False)
    df = df.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)
    df.insert(0, "id", df.index)
    df.to_csv(args.output, index=False)
    print(f"vocab: {len(df):,} term | {args.output}")


if __name__ == "__main__":
    main()