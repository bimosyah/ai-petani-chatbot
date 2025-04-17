import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY_GROQ")
MODEL = os.getenv("LLM_MODEL")


def answer_question(user_question: str) -> str:
    prompt = f"Kamu adalah asisten petani. Jawab pertanyaan berikut dengan jelas dan singkat:\n\n{user_question}\n\nJawaban:"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten pertanian untuk petani Indonesia."},
            {"role": "user", "content": user_question}
        ]
    }

    try:
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=30)
        print("DEBUG:", response.json())

        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {e}"
