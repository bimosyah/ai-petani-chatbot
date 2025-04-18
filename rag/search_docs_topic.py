# rag/search_docs_topic.py
import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

embedding = OpenAIEmbeddings()

ALLOWED_TOPICS = ["jagung", "padi"]


# Fungsi untuk trim berdasarkan jumlah token sederhana
def trim_chunks_to_token_limit(docs, token_limit=3500):
    total_tokens = 0
    selected = []
    for doc in docs:
        token_count = len(doc.page_content.split())  # bisa ganti pakai tiktoken untuk presisi
        if total_tokens + token_count > token_limit:
            break
        selected.append(doc)
        total_tokens += token_count
    return selected

def search_context(question: str, topic: str = "jagung", top_k: int = 8) -> str:
    if topic not in ALLOWED_TOPICS:
        return f"Topik '{topic}' tidak tersedia. Saat ini hanya mendukung topik: jagung dan padi."

    db_path = f"rag/chroma_db/{topic}"
    if not os.path.exists(db_path):
        return f"Vector database untuk topik '{topic}' belum tersedia."

    db = Chroma(persist_directory=db_path, embedding_function=embedding)

    try:
        # Gunakan MMR untuk hasil yang lebih variatif
        results = db.max_marginal_relevance_search(question, k=top_k, fetch_k=20)
    except Exception as e:
        return f"Terjadi kesalahan saat mencari konteks: {e}"

    if not results:
        return "Maaf, tidak ditemukan informasi yang relevan di dokumen."

    trimmed = trim_chunks_to_token_limit(results, token_limit=3500)

    context = "\n\n".join([doc.page_content for doc in trimmed])
    return context

