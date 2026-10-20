import os

from playwright.sync_api import sync_playwright

PDF_DOWNLOAD_FOLDER = "data/pdf"

def download_pdfs(links: list[str]):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page()
        
        for link in links:
            if os.path.exists(f"{PDF_DOWNLOAD_FOLDER}/{link.split('/')[-1]}.pdf"):
                continue
            page.goto(link)
            pdf_url = page.url

            response = page.request.get(pdf_url)
            with open(f"{PDF_DOWNLOAD_FOLDER}/{link.split('/')[-1]}.pdf", "wb") as f:
                f.write(response.body())

            print("PDF berhasil disimpan.")
        browser.close()
        
if __name__=="__main__":
    links = [
        "https://arxiv.org/pdf/1501.05917v1",
        "https://arxiv.org/pdf/1501.00653v1",
        "https://arxiv.org/pdf/1501.02560v1"
    ]
    download_pdfs(links)