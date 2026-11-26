import os
import glob
import time
import io
import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from pypdf import PdfReader
from playwright.sync_api import sync_playwright

def sanitize_data(data):
    """Recursively removes or ignores invalid surrogate characters from strings, dicts, or lists."""
    if isinstance(data, str):
        return data.encode('utf-8', errors='ignore').decode('utf-8')
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data

# ==============================================================================
# 1. KONFIGURASI REGEX FONT BOLD, POLA HEADING, DAN KAMUS SECTION BM25F
# ==============================================================================
BOLD_FONT_PATTERN = re.compile(
    r"(bold|black|heavy|-bd|demi|medi|cmbx|dcbx|ecbx|lmbx|ptmb|phvb|pcrb)",
    re.IGNORECASE
)

HEADING_PATTERN = re.compile(
    r"^("
    # Pola Angka Romawi (I., II., IV., IV.A., III-B., dll)
    r"(?:M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))"
    r"(?:[\.\-](?:[A-Z0-9]+))*[\.\:\)]?\s+[A-Z][A-Za-z0-9\s\-\:\,\(\)&]{2,85}"
    r"|"
    # Pola Angka Desimal Bab/Subbab (1., 1.1, 2.3.4)
    r"\d+(?:\.\d+)*[\.\:\)]?\s+[A-Z][A-Za-z0-9\s\-\:\,\(\)&]{2,85}"
    r"|"
    # Pola Subbab Huruf (A., B., C.)
    r"[A-Z][\.\)]\s+[A-Z][A-Za-z0-9\s\-\:\,\(\)&]{2,85}"
    r"|"
    # Kata kunci section standar paper ilmiah
    r"(?:ABSTRACT|INTRODUCTION|RELATED\s+WORKS?|BACKGROUND|PRELIMINARIES|"
    r"METHODOLOGY|METHODS?|APPROACH|EXPERIMENTS?|EXPERIMENTAL\s+RESULTS?|"
    r"RESULTS?|EVALUATION|DISCUSSION|CONCLUSIONS?|SUMMARY|"
    r"ACKNOWLEDGMENT[S]?|ACKNOWLEDGEMENTS?|REFERENCES|BIBLIOGRAPHY)"
    r")$",
    re.IGNORECASE
)

TARGET_KEYWORDS_LABELS = ["keywords", "key words", "index terms"]

SECTION_LEXICON = {
    "ignore_intro": [
        "introduction", "related work", "related works", "preliminaries",
        "background", "notation", "notations", "motivation", "overview",
        "organization", "problem statement"
    ],
    "metode": [
        "method", "methods", "methodology", "approach", "proposed method",
        "proposed approach", "framework", "system model", "architecture",
        "model", "models", "algorithm", "algorithms", "problem formulation",
        "formulation", "design", "implementation", "technique", "techniques",
        "learning", "training", "theoretic", "theory"
    ],
    "evaluasi": [
        "evaluation", "experiments", "experimental setup", "experimental results",
        "experiment", "results", "simulation", "simulation results",
        "numerical results", "numerical examples", "performance evaluation",
        "performance", "datasets", "dataset", "data", "analysis",
        "empirical", "benchmark", "case study", "validation", "testing"
    ],
    "kesimpulan": [
        "conclusion", "conclusions", "discussion", "discussions",
        "summary", "concluding remarks", "future work", "limitations"
    ],
    "stop": [
        "references", "bibliographical references", "bibliography",
        "acknowledgments", "acknowledgements", "appendix", "proofs"
    ]
}

NOISE_TOKENS = {
    "if", "then", "else", "for", "foreach", "while", "do", "end for",
    "end if", "return", "input", "output", "procedure", "repeat", "until",
    "initialize", "initialization", "parameters", "parameter"
}

# Daftar kata/akronim pendek valid agar tidak ikut tergabung saat perbaikan spasi
VALID_SHORT_UPPER = {
    "OF", "IN", "ON", "TO", "BY", "AS", "OR", "AN", "AT", "IS", "IT", "NO",
    "SO", "UP", "WE", "IF", "DO", "GO", "MY", "US", "BE", "HE", "ME",
    "AI", "ML", "DL", "RL", "IR", "QA", "NLP", "CV", "3D", "2D", "RF", "GP",
    "EM", "KL", "LP", "AR", "VR", "MR", "CT", "UI", "UX", "OS", "IP", "ID",
    "DB", "NN", "GA", "SA", "MC", "EH", "TS", "MP", "DS", "AL", "PH",
    "AF", "II", "IV", "VI", "IX", "XI"
}

VALID_SHORT_LOWER = {
    "of", "in", "on", "to", "by", "as", "or", "an", "at", "is", "it", "no",
    "so", "up", "we", "if", "do", "go", "my", "us", "be", "he", "me"
}


# ==============================================================================
# 2. NORMALISASI KATA TERPISAH & KEMIRIPAN (> 70%)
# ==============================================================================
def normalize_split_words(text: str) -> str:
    """Menyatukan kata yang terpisah akibat spasi OCR/PDF tanpa merusak kata valid."""
    text = unicodedata.normalize("NFKC", text)
    # 1. Gabungkan pemotongan tanda hubung baris: "mod- eling" -> "modeling"
    text = re.sub(r"([a-zA-Z])-\s+([a-zA-Z])", r"\1\2", text)
    # 2. Gabungkan 1 huruf kapital di awal kata: "I NTRODUCTION" -> "INTRODUCTION"
    text = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", text)
    # 3. Gabungkan 2 huruf kapital terpisah jika bukan kata/akronim valid: "EV ALUATION" -> "EVALUATION"
    text = re.sub(
        r"\b([A-Z]{2})\s+([A-Z]{2,})\b",
        lambda m: m.group(0) if m.group(1) in VALID_SHORT_UPPER else f"{m.group(1)}{m.group(2)}",
        text,
    )
    # 4. Gabungkan 1 huruf awal/akhir kata kecil ("T able" -> "Table", "Keyword s" -> "Keywords")
    text = re.sub(r"\b([B-HJ-Zb-hj-z])\s+([a-z]{2,})\b", r"\1\2", text)
    text = re.sub(r"\b([A-Za-z]{3,})\s+([b-hj-z])\b", r"\1\2", text)
    # 5. Gabungkan 2 huruf awal + huruf kecil jika bukan kata pendek valid
    text = re.sub(
        r"\b([A-Za-z][a-z])\s+([a-z]{2,})\b",
        lambda m: m.group(0) if m.group(1).lower() in VALID_SHORT_LOWER else f"{m.group(1)}{m.group(2)}",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def similarity_score(a: str, b: str) -> float:
    """Menghitung rasio kemiripan (0.0 - 1.0) termasuk perbandingan tanpa spasi."""
    a_clean = re.sub(r"[^a-z0-9\s]", "", a.lower()).strip()
    b_clean = re.sub(r"[^a-z0-9\s]", "", b.lower()).strip()
    if not a_clean or not b_clean:
        return 0.0

    ratio_normal = SequenceMatcher(None, a_clean, b_clean).ratio()
    ratio_nospace = SequenceMatcher(
        None, a_clean.replace(" ", ""), b_clean.replace(" ", "")
    ).ratio()
    return max(ratio_normal, ratio_nospace)


def is_similar_to_any(text: str, candidates: list[str], threshold: float = 0.70) -> bool:
    """Mengembalikan True jika kemiripan dengan salah satu kandidat > threshold (70%)."""
    return any(similarity_score(text, cand) > threshold for cand in candidates)


def is_heading_line(line: str) -> bool:
    """Mengecek apakah satu baris teks cocok dengan pola bab/subbab."""
    cleaned = normalize_split_words(line)
    if len(cleaned) < 3 or len(cleaned) > 100:
        return False
    if cleaned.endswith(".") and len(cleaned.split()) > 6:
        return False
    return bool(HEADING_PATTERN.match(cleaned))


def classify_section_header(norm_line: str, is_bold: bool, threshold: float = 0.70) -> list[str]:
    """Mendeteksi apakah baris merupakan header bab/subbab dan mengembalikan kategorinya."""
    if not (is_bold or is_heading_line(norm_line)):
        return []
    if len(norm_line) > 95:
        return []
    if norm_line.endswith(".") and len(norm_line.split()) > 7:
        return []

    clean_title = re.sub(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", "", norm_line).lower()
    clean_title = re.sub(r"[^a-z0-9\s&]", " ", clean_title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()

    if not clean_title or len(clean_title) < 3 or clean_title in NOISE_TOKENS:
        return []

    sub_parts = [p.strip() for p in re.split(r"\b(?:and|&)\b", clean_title) if p.strip()]
    matched_categories = []

    for part in sub_parts:
        for category, vocab_list in SECTION_LEXICON.items():
            if is_similar_to_any(part, vocab_list, threshold=threshold):
                if category not in matched_categories:
                    matched_categories.append(category)

    return matched_categories


def check_keywords_line(norm_line: str, threshold: float = 0.70) -> tuple[bool, str]:
    """
    Mengecek apakah baris mengandung 'keywords', 'key words', atau 'index terms'
    (termasuk kata terpisah dengan kemiripan > 70%).
    """
    item_lower = norm_line.lower()
    item_nospace = item_lower.replace(" ", "")

    if "keyword-based" in item_lower or "keyword based" in item_lower:
        return False, ""

    has_kw_exact = (
        "keywords" in item_lower
        or "key words" in item_lower
        or "index terms" in item_lower
        or "keywords" in item_nospace
        or "indexterms" in item_nospace
    )

    prefix_candidate = re.split(r"[:—–\-]", item_lower)[0].strip()
    has_kw_fuzzy = (
        len(prefix_candidate) <= 25
        and is_similar_to_any(prefix_candidate, TARGET_KEYWORDS_LABELS, threshold=threshold)
    )

    if not (has_kw_exact or has_kw_fuzzy):
        return False, ""

    inline_match = re.search(
        r"(?:k\s*e\s*y\s*w\s*o\s*r\s*d\s*s?|i\s*n\s*d\s*e\s*x\s*t\s*e\s*r\s*m\s*s?)\s*[:—–\-]?\s*(.*)",
        norm_line,
        flags=re.IGNORECASE,
    )
    inline_text = inline_match.group(1).strip() if inline_match else ""
    return True, inline_text


# ==============================================================================
# 3. EKSTRAKSI LANGSUNG DARI PDF BYTES (KEYWORDS + METODE + EVALUASI + KESIMPULAN)
# ==============================================================================
def extract_bm25f_from_pdf_bytes(pdf_bytes: bytes, threshold: float = 0.70) -> dict:
    """
    Membaca file PDF secara langsung menggunakan PdfReader, mendeteksi font bold &
    pola heading, lalu mengekstrak field BM25F:
      - keywords
      - metode
      - evaluasi
      - kesimpulan
      - bold sentences
    """
    empty_result = {
        "keywords": "",
        "metode": "",
        "evaluasi": "",
        "kesimpulan": "",
        "bold sentences": [],
    }

    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        print(" -> Peringatan: Konten bukan file PDF valid (kemungkinan HTML/rate-limit).")
        return empty_result

    reader = PdfReader(io.BytesIO(pdf_bytes))
    structured_lines = []  # (line_text, has_bold)
    bold_items = []
    seen_bold = set()

    for page in reader.pages:
        line_buf = []
        line_has_bold = False
        current_bold_span = []

        def flush_line():
            nonlocal line_buf, line_has_bold
            full_line = "".join(line_buf).strip()
            if full_line:
                structured_lines.append((full_line, line_has_bold))
            line_buf = []
            line_has_bold = False

        def flush_bold_span():
            nonlocal current_bold_span
            if current_bold_span:
                b_txt = re.sub(r"\s+", " ", " ".join(current_bold_span)).strip()
                if b_txt and b_txt not in seen_bold:
                    seen_bold.add(b_txt)
                    bold_items.append(b_txt)
                current_bold_span = []

        def visitor_body(text, cm, tm, font_dict, font_size):
            nonlocal line_has_bold
            if not text:
                return

            font_name = ""
            if font_dict:
                base_font = str(font_dict.get("/BaseFont", ""))
                sub_font = str(font_dict.get("/FontName", ""))
                font_name = f"{base_font} {sub_font}"

            is_bold = bool(BOLD_FONT_PATTERN.search(font_name))

            parts = text.split("\n")
            for idx_p, part in enumerate(parts):
                if part:
                    line_buf.append(part)
                    if is_bold and part.strip():
                        line_has_bold = True
                        current_bold_span.append(part.strip())
                    elif not is_bold and part.strip():
                        flush_bold_span()
                if idx_p < len(parts) - 1:
                    flush_line()
                    flush_bold_span()

        page.extract_text(visitor_text=visitor_body)
        flush_line()
        flush_bold_span()

    # Pelacakan state section di seluruh dokumen PDF
    keywords_list = []
    sections = {"metode": [], "evaluasi": [], "kesimpulan": []}
    current_states = []
    in_keywords_block = False

    for raw_line, is_bold in structured_lines:
        norm_line = normalize_split_words(raw_line)
        if not norm_line:
            continue

        # 1. Cek transisi bab/subbab baru
        detected_cats = classify_section_header(norm_line, is_bold=is_bold, threshold=threshold)
        if detected_cats:
            in_keywords_block = False
            if "stop" in detected_cats:
                break
            if "ignore_intro" in detected_cats and len(detected_cats) == 1:
                current_states = []
                continue

            active = [c for c in detected_cats if c in sections]
            current_states = active
            for cat in current_states:
                sections[cat].append(norm_line)
            continue

        # 2. Cek apakah baris adalah label 'keywords' / 'key words' / 'index terms'
        is_kw, inline_kw = check_keywords_line(norm_line, threshold=threshold)
        if is_kw:
            in_keywords_block = True
            if inline_kw:
                keywords_list.append(inline_kw)
            continue

        # 3. Tangkap baris lanjutan keywords (jika isi keywords di baris bawah label & tidak bold)
        if in_keywords_block:
            if is_heading_line(norm_line) or len(keywords_list) >= 4:
                in_keywords_block = False
            else:
                keywords_list.append(norm_line)
                continue

        # 4. Masukkan seluruh isi paragraf ke dalam section yang sedang aktif
        for cat in current_states:
            sections[cat].append(norm_line)

    def clean_join(items: list[str]) -> str:
        return re.sub(r"\s+", " ", " ".join(items)).strip()

    return {
        "keywords": clean_join(keywords_list).rstrip("."),
        "metode": clean_join(sections["metode"]),
        "evaluasi": clean_join(sections["evaluasi"]),
        "kesimpulan": clean_join(sections["kesimpulan"]),
        "bold sentences": bold_items,
    }


# ==============================================================================
# 4. PIPELINE CRAWLING PLAYWRIGHT & CHECKPOINT PARQUET
# ==============================================================================
SAVE_PATH = "data/papers_2015_2025_structure.parquet"
CHECKPOINT_DIR = "checkpoints"


def start_structure_extraction(parquet_path: str, batch: int = 50, threshold: float = 0.70):
    print("Start direct PDF extraction for BM25F fields (keywords, metode, evaluasi, kesimpulan)...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    links: pd.Series = df["id"]

    # Cek batch checkpoint yang sudah tersimpan
    existing_batches = set()
    for filepath in glob.glob(os.path.join(CHECKPOINT_DIR, "batch_*.parquet")):
        try:
            filename = os.path.basename(filepath)
            batch_num = int(filename.replace("batch_", "").replace(".parquet", ""))
            existing_batches.add(batch_num)
        except ValueError:
            pass

    print(f"Batch yang sudah selesai tersimpan: {sorted(list(existing_batches))}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page()

        current_batch_data = []

        for i, link in enumerate(links):
            current_batch_num = i // batch
            if current_batch_num in existing_batches:
                continue

            link_pdf = link.replace("abs", "pdf")
            extracted = {
                "keywords": "",
                "metode": "",
                "evaluasi": "",
                "kesimpulan": "",
                "bold sentences": [],
            }

            try:
                response = page.request.get(link_pdf, timeout=60000)
                extracted = extract_bm25f_from_pdf_bytes(response.body(), threshold=threshold)
            except Exception as e:
                print(f"Error at index {i} ({link}): {e}")

            current_batch_data.append({
                "id": link,
                "keywords": extracted["keywords"],
                "metode": extracted["metode"],
                "evaluasi": extracted["evaluasi"],
                "kesimpulan": extracted["kesimpulan"],
                "bold sentences": extracted["bold sentences"],
            })

            is_batch_full = len(current_batch_data) == batch
            is_last_item = (i == len(links) - 1)

            if is_batch_full or is_last_item:
                if current_batch_data:
                    cleaned_batch_data = sanitize_data(current_batch_data)
                    batch_df = pd.DataFrame(cleaned_batch_data)
                    batch_file_path = os.path.join(CHECKPOINT_DIR, f"batch_{current_batch_num}.parquet")
                    batch_df.to_parquet(batch_file_path, engine="pyarrow")

                    print(f"Saved batch checkpoint: {batch_file_path}")
                    existing_batches.add(current_batch_num)
                    current_batch_data = []

            # time.sleep(1)

        browser.close()

    combine()


def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")


def combine():
    print("Menggabungkan seluruh batch parquet...")
    all_batch_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "batch_*.parquet")))
    if all_batch_files:
        df_final = pd.concat(
            [pd.read_parquet(f, engine="pyarrow") for f in all_batch_files],
            ignore_index=True
        )
        df_final.to_parquet(SAVE_PATH, engine="pyarrow")
        print(f"Selesai! Hasil akhir disimpan ke: {SAVE_PATH} (Total: {len(df_final)} baris)")
    else:
        print("Tidak ada data baru yang diproses.")
    print("Done")


if __name__ == "__main__":
    # Jalankan ekstraksi langsung dari PDF:
    start_structure_extraction('data/completed.parquet', batch=20, threshold=0.70)
    # combine()