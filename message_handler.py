import json
import os

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


def is_disease_question(question: str) -> bool:
    q = question.lower()
    disease_keywords = [
        "penyakit", "ciri-ciri penyakit", "terinfeksi",
        "gejala", "infeksi", "tanda-tanda"
    ]
    return any(keyword in q for keyword in disease_keywords)


def get_disease_images(topic: str) -> list:
    disease_images = {
        "padi": [
            {"nama": "Busuk Pelepah", "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_4094a5c0-f162-4458-90db-e810ec05b193"},
            {"nama": "Blas", "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_adfbedf5-bc38-4f5f-9ffe-2f5f545c7a1b"},
            {"nama": "Hawar Daun", "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_41080547-d8b3-42dd-8088-3bab8aedd0e8"},
            {"nama": "Bercak Coklat", "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_6ddaf0a5-8813-4066-af4c-f0d207cc4462"},
        ],
        "jagung": [
            {"nama": "Bulai",
             "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_61fe360f-46c8-46a9-aac0-9a3844adddb0"},
            {"nama": "Karat Daun",
             "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_c723467d-a24f-4143-85b7-294b22ecabb7"},
            {"nama": "Hawar Daun",
             "url": "https://s3-ap-southeast-1.amazonaws.com/maxxi-staging/image/user/AVATAR_9bcb6876-fe65-4773-bd73-ce1625fc247f"},
        ]
    }
    return disease_images.get(topic, [])


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
            answer = data["choices"][0]["message"]["content"]

            # Tambahkan gambar jika pertanyaan berkaitan dengan penyakit
            if is_disease_question(user_question):
                images = get_disease_images(topic)
                image_lines = "\n".join([f"- {img['nama']}: {img['url']}" for img in images])
                answer += "\n\n🖼️ Berikut gambar penyakit terkait:\n" + image_lines

            return answer

        return "Tidak ada jawaban dari model."
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {e}"
