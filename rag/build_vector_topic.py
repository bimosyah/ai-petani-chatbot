# rag/build_vector_topic.py
import pickle
import sys

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

if len(sys.argv) < 2:
    print("❗ Masukkan nama topik. Contoh: python build_vector_topic.py padi")
    sys.exit()

topic = sys.argv[1]
input_path = f"rag/chunks/split_docs_{topic}.pkl"
output_path = f"rag/chroma_db/{topic}"

with open(input_path, "rb") as f:
    split_docs = pickle.load(f)

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(split_docs, embedding, persist_directory=output_path)
db.persist()

print(f"✅ Vector DB untuk topik '{topic}' berhasil disimpan di {output_path}/")
