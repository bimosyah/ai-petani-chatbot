# rag/search_docs_topic.py
import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

embedding = OpenAIEmbeddings()

ALLOWED_TOPICS = ["jagung", "padi"]


def search_context(question: str, topic: str = "jagung", top_k: int = 3) -> str:
    if topic not in ALLOWED_TOPICS:
        return f"Topik '{topic}' tidak tersedia. Saat ini hanya mendukung topik: jagung dan padi."

    db_path = f"rag/chroma_db/{topic}"
    if not os.path.exists(db_path):
        return f"Vector database untuk topik '{topic}' belum tersedia."

    db = Chroma(persist_directory=db_path, embedding_function=embedding)
    results = db.similarity_search(question, k=top_k)

    if not results:
        return "Maaf, tidak ditemukan informasi yang relevan di dokumen."

    context = "\n\n".join([doc.page_content for doc in results])
    return context
