"""
Download LegalBench-RAG from Dropbox and unzip it under data/raw/. Run from the repo root on Kaggle/Colab
(internet must be ON). The loader in src/data.py finds the 'corpus' and 'benchmarks' folders automatically.

If this fails (Dropbox sometimes blocks scripts): download the zip in your browser from the link in the
LegalBench-RAG GitHub README (github.com/zeroentropy-cc/legalbenchrag), upload it as a Kaggle Dataset, and
skip this step. configs/config.yaml already searches /kaggle/input.
"""
import os
import zipfile

import requests

URL = ("https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
       "?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1")
OUT = "data/raw"
ZIP = os.path.join(OUT, "legalbench_rag.zip")
os.makedirs(OUT, exist_ok=True)

print("Downloading (this can take a few minutes)...")
r = requests.get(URL, stream=True, timeout=600)
r.raise_for_status()
with open(ZIP, "wb") as f:
    for chunk in r.iter_content(chunk_size=1 << 20):
        f.write(chunk)
print(f"Downloaded {os.path.getsize(ZIP) / 1e6:.1f} MB")

with zipfile.ZipFile(ZIP) as zf:
    zf.extractall(os.path.join(OUT, "legalbench_rag"))
os.remove(ZIP)

for dirpath, dirnames, filenames in os.walk(os.path.join(OUT, "legalbench_rag")):
    depth = dirpath.count(os.sep) - OUT.count(os.sep)
    if depth <= 3:
        print(dirpath, "->", len(filenames), "files")
print("Done. Next: python scripts/01_prepare_data.py")
