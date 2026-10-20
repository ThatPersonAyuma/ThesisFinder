import os
import glob
import pandas as pd
import numpy as np
from playwright.sync_api import sync_playwright

from downloader import PDF_DOWNLOAD_FOLDER
from extractor import extract_keywords_from_bytes

SAVE_PATH = 'data/papers_2015_2025_keywords.parquet'
KEYWORDS_COL_NAME='keywords'
CHECKPOINT_DIR = "checkpoints"

def start_keywords_extraction(parquet_path: str, batch=50):
    print('start getting keywords')
    
    # Buat folder khusus untuk simpan file parquet terpisah jika belum ada
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    links: pd.Series[str] = df['id']

    # 1. CEK RESUME: Cari batch yang sudah pernah disimpan sebelumnya
    existing_batches = set()
    for filepath in glob.glob(os.path.join(CHECKPOINT_DIR, "batch_*.parquet")):
        try:
            # Mengambil angka batch dari nama file (misal: "batch_2.parquet" -> 2)
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

            # Lewati batch jika file parquet batch tersebut sudah ada
            if current_batch_num in existing_batches:
                continue

            link_pdf = link.replace('abs', 'pdf')
            keyword_res = None

            try:
                page.goto(link_pdf)
                pdf_url = page.url
                response = page.request.get(pdf_url, timeout=60000)
                keyword_res = extract_keywords_from_bytes(response.body())
            except Exception as e:
                print(f"error at index {i} ({link}): {e}")

            # Tampung hasil ekstraksi data batch saat ini
            current_batch_data.append({
                'id': link,
                KEYWORDS_COL_NAME: keyword_res
            })

            # Simpan ke parquet terpisah jika batch terpenuhi ATAU sudah di data terakhir
            is_batch_full = len(current_batch_data) == batch
            is_last_item = (i == len(links) - 1)

            if is_batch_full or is_last_item:
                if current_batch_data:
                    batch_df = pd.DataFrame(current_batch_data)
                    batch_file_path = os.path.join(CHECKPOINT_DIR, f"batch_{current_batch_num}.parquet")
                    batch_df.to_parquet(batch_file_path, engine='pyarrow')
                    
                    print(f"Saved batch checkpoint: {batch_file_path}")
                    existing_batches.add(current_batch_num)
                    
                    # Reset penampung data untuk batch berikutnya
                    current_batch_data = []

        browser.close()

    # 2. GABUNGKAN SEMUA BATCH: Gabung parquet terpisah ke SAVE_PATH di akhir proses
    print("Menggabungkan seluruh batch parquet...")
    all_batch_files = glob.glob(os.path.join(CHECKPOINT_DIR, "batch_*.parquet"))
    
    if all_batch_files:
        df_keywords = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in all_batch_files], ignore_index=True)
        
        # Gabungkan data kata kunci kembali ke dataframe asal berdasarkan 'id'
        df_final = df.merge(df_keywords[['id', KEYWORDS_COL_NAME]], on='id', how='left')
        df_final.to_parquet(SAVE_PATH, engine='pyarrow')
        print(f"Selesai! Hasil akhir disimpan ke: {SAVE_PATH}")
    else:
        print("Tidak ada data baru yang diproses.")

    print("Done")
    
def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")
        

if __name__ == "__main__":
    start_keywords_extraction('data/papers_2015_2025.parquet', batch=20)