import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]               # project root
MODEL_PATH = ROOT / "models" / "ner_bert_gmb" / "checkpoint-450"

def ensure_model() -> None:
    """If model folder missing/empty, run fetch_model.py with the current Python."""
    if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
        print(f"[INFO] Model found at: {MODEL_PATH}")
        return

    print("[INFO] Model not found. Fetching…")
    fetcher = ROOT / "scripts" / "fetch_model.py"
    subprocess.run([sys.executable, str(fetcher)], check=True)

if __name__ == "__main__":
    ensure_model()