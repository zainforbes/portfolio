import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash-lite")
resp = model.generate_content("Hello Gemini, respond with 'OK' if you're working.")
print(resp.text)
