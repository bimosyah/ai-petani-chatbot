# load_pdf.py
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# 1. Load PDF
loader = PyMuPDFLoader("rag/jagung.pdf")  # ← Ganti nama file jika perlu
documents = loader.load()

# 2. Split jadi chunk kecil
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# 3. Simpan ke file (sementara)
with open("rag/split_docs.pkl", "wb") as f:
    pickle.dump(split_docs, f)

print(f"{len(split_docs)} chunks berhasil dibuat dan disimpan!")
