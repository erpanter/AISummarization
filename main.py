from pathlib import Path
from pipelines.genai_pipeline import GeminiSummarizer
from pipelines.NERPipeline import BertNERPipeline
from utils.text_processing import chunk_by_entities
from utils.prompts import CHUNK_SUMMARY, FINAL_SUMMARY
from scripts.ensure_model import ensure_model
from dotenv import load_dotenv

# Get project root (folder where main.py is located)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
ensure_model()  # Ensure model is downloaded

def predict_sentence_entities_factory(ner: BertNERPipeline):
    """Wraps .predict() to return entity dicts per sentence."""
    def fn(sentence: str):
        return ner.predict(sentence)
    return fn

def summarize_document(doc_path: Path, ner_model_path: Path):
    text = doc_path.read_text(encoding="utf-8")

    # Instantiate NER pipeline
    ner = BertNERPipeline(
        model_name_or_path=str(ner_model_path),
        label2id_path=PROJECT_ROOT / "data" / "labels" / "label2id.json",
        id2label_path=PROJECT_ROOT / "data" / "labels" / "id2label.json"
    )

    # Instantiate Gemini
    gem = GeminiSummarizer(model="gemini-1.5-flash", temperature=0.2)

    # Chunk doc
    chunks = chunk_by_entities(
        text,
        ner_sentence_entities=predict_sentence_entities_factory(ner),
        max_tokens_per_chunk=2200,
        hard_cap=2600
    )

    mini_summaries = [gem.summarize(ch["text"], CHUNK_SUMMARY) for ch in chunks]
    stitched = "\n".join(f"- {s}" for s in mini_summaries if s.strip())
    final = gem.summarize(stitched, FINAL_SUMMARY)

    return final, mini_summaries

if __name__ == "__main__":
    # Paths based on project root
    doc_path = PROJECT_ROOT / "data" / "raw" / "sample_document.txt"
    ner_model_path = PROJECT_ROOT / "models" / "ner_bert_gmb" / "checkpoint-450"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    final, minis = summarize_document(doc_path, ner_model_path)

    (processed_dir / "summary_output.txt").write_text(final, encoding="utf-8")
    (processed_dir / "chunk_summaries.txt").write_text("\n\n".join(minis), encoding="utf-8")

    print("Done. Wrote summary_output.txt + chunk_summaries.txt to", processed_dir)