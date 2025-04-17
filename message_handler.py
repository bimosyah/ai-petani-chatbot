import os
import json
import httpx
from dotenv import load_dotenv

from rag.search_docs import search_context

load_dotenv()

API_KEY = os.getenv("API_KEY_GROQ")
MODEL = os.getenv("LLM_MODEL")

# Load static Q&A cache
with open("cache/static_cache.json") as f:
    static_cache = json.load(f)

def answer_question(user_question: str) -> str:
    normalized = user_question.strip().lower()

    # Cek static cache dulu
    for key in static_cache:
        if key in normalized:
            return static_cache[key]
    # Ambil konteks dari PDF jagung
    context = search_context(user_question)

    prompt = (
        "Gunakan informasi berikut untuk menjawab pertanyaan petani.\n\n"
        f"{context}\n\n"
        f"Pertanyaan: {user_question}\n\n"
        "Jawaban:"
    )

    # Kirim prompt ke LLM
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten pertanian untuk petani Indonesia."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=30)
        data = response.json()

        if "error" in data:
            return f"LLM error: {data['error']['message']}"

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]

        return "Tidak ada jawaban dari model."
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {e}"
