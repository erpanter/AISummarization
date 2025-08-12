from pathlib import Path
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------- project paths / env ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")  # GEMINI_API_KEY, MODEL_ZIP_URL, etc.

# NEW: ensure model is present before importing/instantiating NER
from scripts.ensure_model import ensure_model
ensure_model()  # downloads + unzips if models/ner_bert_gmb/checkpoint-450 is missing

# now safe to import pipelines
from pipelines.NERPipeline import BertNERPipeline
from pipelines.genai_pipeline import GeminiSummarizer
from utils.text_processing import chunk_by_entities
from utils.prompts import CHUNK_SUMMARY, FINAL_SUMMARY
from app.extract_text import extract_text

app = FastAPI(title="AI Summarizer (BERT NER → Gemini)")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- singletons ----------
NER_MODEL_PATH = PROJECT_ROOT / "models" / "ner_bert_gmb" / "checkpoint-450"
LABEL2ID = PROJECT_ROOT / "data" / "labels" / "label2id.json"
ID2LABEL = PROJECT_ROOT / "data" / "labels" / "id2label.json"

ner: BertNERPipeline | None = None
gem: GeminiSummarizer | None = None

@app.on_event("startup")
def _startup():
    global ner, gem
    # instantiate NER (model files guaranteed by ensure_model)
    ner = BertNERPipeline(
        model_name_or_path=str(NER_MODEL_PATH),
        label2id_path=str(LABEL2ID),
        id2label_path=str(ID2LABEL),
    )
    # Gemini client
    gem = GeminiSummarizer(model="gemini-1.5-flash", temperature=0.2)

class SummarizeResponse(BaseModel):
    final_summary: str
    chunks: list[str]

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # save temp
    tmp_dir = PROJECT_ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / file.filename
    try:
        with open(dst, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # extract text
    try:
        text = extract_text(dst, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")
    finally:
        try: os.remove(dst)
        except Exception: pass

    if not text or len(text) < 10:
        raise HTTPException(status_code=400, detail="No meaningful text found in the file.")

    assert ner is not None and gem is not None, "Service not initialized"

    def ner_sentence_entities(sentence: str):
        if ner is None:
            raise HTTPException(status_code=500, detail="NER service not initialized")
        return ner.predict(sentence)

    chunks = chunk_by_entities(
        text,
        ner_sentence_entities=ner_sentence_entities,
        max_tokens_per_chunk=2200,
        hard_cap=2600
    )

    mini_summaries: list[str] = []
    for ch in chunks:
        mini_summaries.append(gem.summarize(ch["text"], CHUNK_SUMMARY))

    stitched = "\n".join(f"- {s}" for s in mini_summaries if s.strip())
    final_summary = gem.summarize(stitched, FINAL_SUMMARY)

    return JSONResponse({
        "final_summary": final_summary,
        "chunks": mini_summaries
    })

@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "app" / "static" / "index.html")

app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "app" / "static"),
    name="static",
)
