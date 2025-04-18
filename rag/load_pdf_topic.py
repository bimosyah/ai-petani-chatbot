# rag/load_pdf_topic.py
import pickle
import sys

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if len(sys.argv) < 2:
    print("❗ Masukkan nama file PDF-nya. Contoh: python load_pdf_topic.py jagung")
    sys.exit()

topic = sys.argv[1]
pdf_path = f"rag/files/{topic}.pdf"
output_path = f"rag/chunks/split_docs_{topic}.pkl"

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

with open(output_path, "wb") as f:
    pickle.dump(split_docs, f)

print(f"✅ {len(split_docs)} chunks berhasil dibuat dan disimpan ke {output_path}")
