import os
import json
import httpx
from dotenv import load_dotenv

from rag.search_docs_topic import search_context

load_dotenv()

API_KEY = os.getenv("API_KEY_GROQ")
MODEL = os.getenv("LLM_MODEL")

def detect_topic(question: str) -> str:
    q = question.lower()
    if "jagung" in q:
        return "jagung"
    elif "padi" in q:
        return "padi"
    else:
        return "unknown"

# Load static Q&A cache
with open("cache/static_cache.json") as f:
    static_cache = json.load(f)

def answer_question(user_question: str) -> str:
    normalized = user_question.strip().lower()

    greetings = ["halo", "hai", "assalamualaikum", "selamat pagi", "selamat siang"]
    if any(greet in normalized for greet in greetings):
        return "Halo! 👋 Saya chatbot AI pertanian. Saya bisa bantu soal **padi** dan **jagung**. Silakan tanya ya!"

    # 1. Cek dari static cache dulu
    for key in static_cache:
        if key in normalized:
            return static_cache[key]

    # 2. Deteksi topik pertanyaan
    topic = detect_topic(user_question)
    if topic == "unknown":
        return (
            "Saat ini saya baru bisa bantu untuk topik **padi** dan **jagung** ya 🌾🌽.\n"
            "Coba contoh pertanyaan seperti:\n"
            "- *Apa saja hama jagung?*\n"
            "- *Kapan panen padi?*"
        )

    # 3. Ambil konteks dari PDF topik yang sesuai
    context = search_context(user_question, topic=topic)

    # 4. Jika konteks gagal ditemukan (misal error karena vector DB belum dibuat)
    if context.startswith("Topik") or context.startswith("Vector"):
        return context

    # 5. Buat prompt dan kirim ke LLM
    prompt = (
        f"Gunakan informasi berikut dari dokumen topik **{topic}** untuk menjawab pertanyaan petani.\n\n"
        f"{context}\n\n"
        f"Pertanyaan: {user_question}\n\n"
        "Jawaban:"
    )

    # LLM call (seperti sebelumnya)
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
