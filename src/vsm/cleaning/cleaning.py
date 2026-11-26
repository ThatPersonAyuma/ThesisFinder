import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..tfidf import DATASET_PATH, FIELDS, VSM_DATA_DIR

INPUT = VSM_DATA_DIR / "frek.csv"
OUTPUT = VSM_DATA_DIR / "frek_bersih.csv"
MIN_FREQ = 5

LINK = re.compile(r"https?|www\.|://|github\.(?:com|io)|\.(?:com|org|net|edu|io)\b", re.I)
ANEH = re.compile(
    r"[\\{}$^_=<>|~@#]"      # sisa LaTeX / kode: \emph{..}, $x^2$, snake_case, a=b
    r"|%"                    # persen: 99.90%
    r"|\d[.,]\d"             # angka desimal / ribuan: 4.83x, 1,000
    r"|(.)\1{3,}"            # karakter berulang 4x atau lebih: aaaa, ----
    r"|^[^A-Za-z]*$"         # tidak ada huruf sama sekali: 546, (200, 35,
    r"|[^\x00-\x7f]"         # non-ASCII langsung dibuang pada langkah filter mentah
)
TOKEN = re.compile(r"[a-z0-9]+")


def cocok(df: pd.DataFrame, pola: re.Pattern) -> pd.Series:
    return df["word"].map(lambda w: bool(pola.search(w)))


def read_freq_csv(path: str | Path) -> pd.DataFrame:
    """Baca file frekuensi CSV (mendukung file ber-header maupun tanpa header)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig").iloc[:, :2]
    if len(df) > 0:
        first_freq = pd.to_numeric(df.iloc[0, 1], errors="coerce")
        if pd.notna(first_freq):
            # Baris pertama ternyata data (tanpa header), baca ulang dengan header=None.
            df = pd.read_csv(
                path, header=None, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            ).iloc[:, :2]
    df.columns = ["word", "frequence"]
    df["frequence"] = pd.to_numeric(df["frequence"], errors="coerce")
    df = df.dropna(subset=["frequence"])
    df = df[df["word"] != ""]
    return df.astype({"frequence": int}).reset_index(drop=True)


def build_word_freq_from_parquet(
    parquet_path: str | Path = DATASET_PATH,
    out_path: str | Path = INPUT,
    fields: tuple[str, ...] = FIELDS,
    batch_size: int = 20_000,
    limit: int | None = None,
) -> pd.DataFrame:
    """Ekstrak frekuensi kata mentah dari kolom teks dataset parquet -> frek.csv."""
    pf = pq.ParquetFile(parquet_path)
    cols = [c for c in fields if c in pf.schema_arrow.names]
    if not cols:
        raise ValueError(f"Tidak ada kolom {list(fields)} di {parquet_path}")

    freq: Counter[str] = Counter()
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        if limit is not None and seen + len(df) > limit:
            df = df.iloc[: limit - seen]
        for col in cols:
            for text in df[col].fillna("").astype(str).tolist():
                freq.update(text.split())
        seen += len(df)
        if limit is not None and seen >= limit:
            break

    out = pd.DataFrame(freq.items(), columns=["word", "frequence"])
    out = out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


# 1. Mengubah semua kata jadi huruf kecil (lowercase), frekuensi kata yang sama digabung
def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.assign(word=df["word"].str.lower()).groupby("word", as_index=False)["frequence"].sum()
    return out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)


# 2. Menghilangkan link (http, https, www, github.com, .com, .org, dst.)
def hapus_link(df: pd.DataFrame) -> pd.DataFrame:
    return df[~cocok(df, LINK)]


# 3. Menghilangkan bentuk kata yang tidak masuk akal (regex)
def hapus_bentuk_aneh(df: pd.DataFrame) -> pd.DataFrame:
    return df[~cocok(df, ANEH)]


# 4. Normalisasi (NFKD + regex): buang tanda baca, pecah simbol/strip, gabung frekuensi term sama
def normalisasi(df: pd.DataFrame) -> pd.DataFrame:
    freq: Counter[str] = Counter()
    for word, n in zip(df["word"], df["frequence"]):
        norm = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")
        for t in TOKEN.findall(norm):
            if len(t) > 1:
                freq[t] += n
    out = pd.DataFrame(freq.items(), columns=["word", "frequence"])
    if len(out) == 0:
        return pd.DataFrame(columns=["word", "frequence"])
    return out.sort_values(["frequence", "word"], ascending=[False, True], ignore_index=True)


# 5. Menghilangkan angka murni (hasil pecahan langkah 4, mis. 2024)
def hapus_angka(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df[~df["word"].str.isdigit()]


# 6. Menghilangkan kata yang total frekuensinya setelah normalisasi masih di bawah MIN_FREQ
def hapus_freq_rendah(df: pd.DataFrame, min_freq: int = MIN_FREQ) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df[df["frequence"] >= min_freq].reset_index(drop=True)


def clean_frequencies(df: pd.DataFrame, min_freq: int = MIN_FREQ, verbose: bool = True) -> pd.DataFrame:
    """Jalankan seluruh tahap pembersihan pada DataFrame [word, frequence]."""
    if verbose:
        print(f"awal                          : {len(df):>9,}")
    steps = [
        ("1. lowercase + gabung freq", lowercase),
        ("2. link dibuang", hapus_link),
        ("3. bentuk aneh (regex) dibuang", hapus_bentuk_aneh),
        ("4. normalisasi + gabung freq", normalisasi),
        ("5. angka murni dibuang", hapus_angka),
        (f"6. freq < {min_freq} dibuang", lambda d: hapus_freq_rendah(d, min_freq)),
    ]
    for nama, fungsi in steps:
        df = fungsi(df)
        if verbose:
            print(f"{nama:<30}: {len(df):>9,}")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Bersihkan frek.csv -> frek_bersih.csv")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument("--from-parquet", default=None, help="opsional: ekstrak frek mentah dari parquet dulu")
    args = ap.parse_args()

    if args.from_parquet:
        df = build_word_freq_from_parquet(args.from_parquet, args.input)
    else:
        df = read_freq_csv(args.input)

    df = clean_frequencies(df, min_freq=args.min_freq, verbose=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Tersimpan: {args.output}")


if __name__ == "__main__":
    main()
