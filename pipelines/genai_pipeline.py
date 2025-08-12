import os
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiSummarizer:
    def __init__(self, model="gemini-1.5-flash", temperature=0.2, api_key=None):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def summarize(self, text: str, instructions: str) -> str:
        resp = self.model.generate_content(
            [{"role":"user","parts":[{"text": f"{instructions}\n\n---\n{text}"}]}],
            generation_config={"temperature": self.temperature}
        )
        return getattr(resp, "text", "").strip()
