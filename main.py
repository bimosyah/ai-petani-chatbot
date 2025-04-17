from fastapi import FastAPI
from message_handler import answer_question

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Petani Chatbot"}

@app.post("/ask")
def ask(question: str):
    answer = answer_question(question)
    return {"question": question, "answer": answer}
