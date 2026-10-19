import os

def extract_keywords(immediate_delete:bool=True):
    delete_file()
    pass
    
def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")
