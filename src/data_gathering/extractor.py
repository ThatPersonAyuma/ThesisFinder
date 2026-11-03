# importing required classes
from io import BytesIO

from pypdf import PdfReader
from downloader import PDF_DOWNLOAD_FOLDER
from os import listdir

def extract_keywords(file_path: str, only_first_page: bool = True)->str:
    # creating a pdf reader object
    reader = PdfReader(file_path)

    # Normally keywords in the first page
    pages = [reader.pages[0]] if only_first_page else reader.pages
    keywords = ""
    for page in pages:
        # extracting text from page
        data = page.extract_text().split("\n")
        for item in data:
            item = item.lower()
            if ("keywords" in item 
                    or "key words" in item):
                keywords = item
    return keywords

def extract_keywords_from_bytes(pdf_source:bytes, first_page_only:bool=True) -> str|None:
    reader = PdfReader(BytesIO(pdf_source))

    pages =  [reader.pages[0]] if first_page_only else reader.pages
    keywords = None
    
    for page in pages:
        text = page.extract_text()
        if not text:
            continue
            
        data = text.split("\n")
        for item in data:
            item_lower = item.lower()
            if "keywords" in item_lower or "key words" in item_lower \
                or "index terms" in item_lower:
                keywords = item
                return keywords 

    return keywords

def extract_all():
    files = [f"{PDF_DOWNLOAD_FOLDER}/{file}" for file in listdir(PDF_DOWNLOAD_FOLDER)]
    for file in files:
        print(extract_keywords(file))


if __name__ == "__main__":
    extract_all()