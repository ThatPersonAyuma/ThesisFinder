import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ..tfidf import ROOT_DIR

INPUT = ROOT_DIR / "frek.csv"
OUTPUT = ROOT_DIR / "frek_bersih.csv"
MIN_FREQ = 5


def cocok(df: pd.DataFrame, pola: re.Pattern) -> pd.Series:
    return df["word"].map(lambda w: bool(pola.search(w)))


# 1. Mengubah semua kata jadi huruf kecil (lowercase), frekuensi kata yang sama digabung
def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.assign(word=df["word"].str.lower()).groupby("word", as_index=False)["frequence"].sum()
    return out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)


# 2. Menghilangkan kata yang freq-nya di bawah MIN_FREQ
def hapus_freq_rendah(df: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    return df[df["frequence"] >= min_freq]


# 3. Menghilangkan link (http, https, www, github.com, .com, .org, dst.)
LINK = re.compile(r"https?|www\.|://|github\.(?:com|io)|\.(?:com|org|net|edu|io)\b", re.I)


def hapus_link(df: pd.DataFrame) -> pd.DataFrame:
    return df[~cocok(df, LINK)]


# 4. Menghilangkan bentuk kata yang tidak masuk akal (regex)
ANEH = re.compile(
    r"[\\{}$^_=<>|~@#]"      # sisa LaTeX / kode: \emph{..}, $x^2$, snake_case, a=b
    r"|%"                    # persen: 99.90%
    r"|\d[.,]\d"             # angka desimal / ribuan: 4.83x, 1,000
    r"|(.)\1{3,}"            # karakter berulang 4x atau lebih: aaaa, ----
    r"|^[^A-Za-z]*$"         # tidak ada huruf sama sekali: 546, (200, 35,
    r"|[^\x00-\x7f]"         # huruf non-ASCII (Schrödinger, naïve): akan pecah jadi potongan tak bermakna
)


def hapus_bentuk_aneh(df: pd.DataFrame) -> pd.DataFrame:
    return df[~cocok(df, ANEH)]


# 5. Normalisasi (regex): buang tanda baca, pecah di simbol/strip, gabung frekuensi term yang sama
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def normalisasi(df: pd.DataFrame) -> pd.DataFrame:
    freq = Counter()
    for word, n in zip(df["word"], df["frequence"]):
        for t in TOKEN.findall(word):
            if len(t) > 1:
                freq[t] += n
    out = pd.DataFrame(freq.items(), columns=["word", "frequence"])
    return out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)


# 6. Menghilangkan angka murni (hasil pecahan langkah 5, mis. 2024)
def hapus_angka(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["word"].str.isdigit()]


def main():
    ap = argparse.ArgumentParser(description="Bersihkan frek.csv -> frek_bersih.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig").iloc[:, :2]
    df.columns = ["word", "frequence"]
    df["frequence"] = pd.to_numeric(df["frequence"], errors="coerce")
    df = df.dropna(subset=["frequence"]).astype({"frequence": int})
    print(f"awal                          : {len(df):>9,}")

    for nama, fungsi in [
        ("1. lowercase + gabung freq", lowercase),
        (f"2. freq < {args.min_freq} dibuang", lambda d: hapus_freq_rendah(d, args.min_freq)),
        ("3. link dibuang", hapus_link),
        ("4. bentuk aneh (regex) dibuang", hapus_bentuk_aneh),
        ("5. normalisasi + gabung freq", normalisasi),
        ("6. angka murni dibuang", hapus_angka),
    ]:
        df = fungsi(df)
        print(f"{nama:<30}: {len(df):>9,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Tersimpan: {args.output}")


if __name__ == "__main__":
    main()