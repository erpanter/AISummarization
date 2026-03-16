import os
from dotenv import load_dotenv
import google.generativeai as genai


class GeminiSummarizer:

    def __init__(self, model=None, temperature=0.2, api_key=None):
        load_dotenv()

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)

        # auto-detect model if not specified
        if model is None:
            model = self._detect_model()

        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

        print(f"Using Gemini model: {model}")

    def _detect_model(self):
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        # prefer fast models
        priorities = ["flash", "pro"]

        for p in priorities:
            for m in models:
                if p in m:
                    return m

        return models[0]  # fallback

    def summarize(self, text: str, instructions: str) -> str:
        resp = self.model.generate_content(
            [{"role": "user", "parts": [{"text": f"{instructions}\n\n---\n{text}"}]}],
            generation_config={"temperature": self.temperature}
        )

        return getattr(resp, "text", "").strip()