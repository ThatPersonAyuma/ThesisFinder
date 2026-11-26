import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..tfidf import VOCAB_PATH, VSM_DATA_DIR, analyze_text, query_vector

INPUT = VSM_DATA_DIR / "frek_stem.csv"
CLEAN_PATH = VOCAB_PATH


class Tokenizer:
    """Teks -> token, bag-of-words, atau vektor TF-IDF selaras dengan kosakata VSM."""

    def __init__(self, vocab_path: str | Path | list[str] | dict[str, int] = CLEAN_PATH):
        if isinstance(vocab_path, dict):
            self.source = None
            terms = [t for t, _ in sorted(vocab_path.items(), key=lambda kv: kv[1])]
        elif isinstance(vocab_path, (list, tuple)):
            self.source = None
            terms = list(vocab_path)
        else:
            path = Path(vocab_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Vocab belum ada di {path}. Jalankan: python -m src.vsm.cleaning"
                )
            self.source = path
            if path.suffix.lower() == ".json":
                terms = json.loads(path.read_text(encoding="utf-8"))
            else:
                terms = pd.read_csv(path, dtype={"word": str}, keep_default_na=False)["word"].tolist()

        self.terms = terms
        self.vocab = {t: i for i, t in enumerate(self.terms)}

    def __len__(self) -> int:
        return len(self.terms)

    def tokenize(self, text: str) -> list[str]:
        return [t for t in analyze_text(text) if t in self.vocab]

    def encode(self, text: str) -> Counter:
        return Counter(self.vocab[t] for t in self.tokenize(text))

    def vectorize(self, text: str, sparse: bool = False):
        """Teks -> vektor bag-of-words (jumlah kemunculan tiap term) berukuran len(self)."""
        enc = self.encode(text)
        cols = np.fromiter(enc.keys(), dtype=np.int64, count=len(enc))
        vals = np.fromiter(enc.values(), dtype=np.int32, count=len(enc))
        if sparse:
            from scipy.sparse import csr_matrix

            return csr_matrix((vals, (np.zeros(len(cols), dtype=np.int64), cols)), shape=(1, len(self)))
        vec = np.zeros(len(self), dtype=np.int32)
        if len(cols) > 0:
            vec[cols] = vals
        return vec

    def vectorize_tfidf(self, text: str, idf: np.ndarray, sparse: bool = False):
        """Teks -> vektor TF-IDF (sublinear tf * idf, L2-normalized) berukuran len(self)."""
        enc = self.encode(text)
        ids, weights = query_vector(enc, idf)
        if sparse:
            from scipy.sparse import csr_matrix

            return csr_matrix(
                (weights.astype(np.float32), (np.zeros(len(ids), dtype=np.int64), ids)),
                shape=(1, len(self)),
            )
        vec = np.zeros(len(self), dtype=np.float32)
        if len(ids) > 0:
            vec[ids] = weights.astype(np.float32)
        return vec


def build_vocab_csv(df_stem: pd.DataFrame) -> pd.DataFrame:
    """Susun DataFrame vocab_clean [id, word, frequence] dari hasil stemming."""
    if len(df_stem) == 0:
        return pd.DataFrame(columns=["id", "word", "frequence"])
    out = df_stem.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True).copy()
    out.insert(0, "id", out.index)
    return out


def main():
    ap = argparse.ArgumentParser(description="frek_stem.csv -> vocab_clean.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(CLEAN_PATH))
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"word": str}, keep_default_na=False)
    df = build_vocab_csv(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"vocab: {len(df):,} term | {args.output}")


if __name__ == "__main__":
    main()
