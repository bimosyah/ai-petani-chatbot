# rag/build_vector_db.py
import pickle

from langchain.embeddings import OpenAIEmbeddings  # atau bisa diganti HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load hasil split
with open("rag/split_docs.pkl", "rb") as f:
    split_docs = pickle.load(f)

# 2. Gunakan OpenAI Embeddings (butuh OPENAI_API_KEY di .env)
embedding = OpenAIEmbeddings()

# 3. Build Chroma vectorstore
db = Chroma.from_documents(split_docs, embedding, persist_directory="rag/chroma_db")

# 4. Simpan vector DB
db.persist()

print("✅ Chroma vector DB berhasil dibuat dan disimpan di rag/chroma_db/")
