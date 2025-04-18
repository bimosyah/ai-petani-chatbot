# sheet_logger.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Scope & Auth
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("ai_petani_chatbot.json", scope)
client = gspread.authorize(creds)

# Buka Sheet
sheet = client.open_by_key("1UXAxmauHqD6CSoRfPfjpsvjRpqBRh6OtbZj2khA5KlU").sheet1

def log_to_sheet(question: str, answer: str, topic: str, source: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, question, topic, source, answer]
    sheet.append_row(row)
