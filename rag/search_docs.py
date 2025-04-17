from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# Load vector DB
embedding = OpenAIEmbeddings()
db = Chroma(persist_directory="rag/chroma_db", embedding_function=embedding)


def search_context(question: str, top_k: int = 3) -> str:
    results = db.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context
