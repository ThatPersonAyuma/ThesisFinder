# importing required classes
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

def extract_all():
    files = [f"{PDF_DOWNLOAD_FOLDER}/{file}" for file in listdir(PDF_DOWNLOAD_FOLDER)]
    for file in files:
        print(extract_keywords(file))


if __name__ == "__main__":
    extract_all()