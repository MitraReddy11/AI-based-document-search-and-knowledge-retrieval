import os
import shutil

def startup_cleanup():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Old vector DB removed.")
    else:
        print("No old DB found.")