# scripts/fetch_model.py
from __future__ import annotations
from pathlib import Path
import os, zipfile, shutil, hashlib, time
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
DEST       = MODELS_DIR / "ner_bert_gmb" / "checkpoint-450"
ZIP_PATH   = MODELS_DIR / "model.zip"

MODEL_ZIP_URL    = (os.getenv("MODEL_ZIP_URL") or "").strip()
MODEL_ZIP_SHA256 = (os.getenv("MODEL_ZIP_SHA256") or "").strip()

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_http(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading from {url} …")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0; start = time.time()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk: continue
                f.write(chunk); done += len(chunk)
                if total:
                    pct = done * 100 // total
                    speed = done / max(1, time.time()-start) / (1024*1024)
                    print(f"\rDownloading: {pct:3d}% {done/1e6:,.1f}/{total/1e6:,.1f} MB ({speed:.1f} MB/s)", end="")
    print()

def unzip_and_flatten(zip_path: Path):
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit("Downloaded file is not a valid ZIP. Check MODEL_ZIP_URL.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODELS_DIR)

    # If the expected path already exists, we're done
    if DEST.exists() and any(DEST.iterdir()):
        return

    # Search for any 'checkpoint-450' folder inside models/
    candidates = list(MODELS_DIR.rglob("checkpoint-450"))
    if not candidates:
        raise SystemExit("Couldn't find 'checkpoint-450' after unzip. Check ZIP contents.")
    # Pick the shortest path (closest match)
    src = sorted(candidates, key=lambda p: len(p.parts))[0]

    # Ensure target parent exists
    DEST.parent.mkdir(parents=True, exist_ok=True)

    # If src is already DEST, done
    if src.resolve() == DEST.resolve():
        pass
    else:
        # Move everything from src into DEST
        if not DEST.exists():
            DEST.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.move(str(item), str(DEST))
        # Clean up the now-empty src tree if possible
        try:
            shutil.rmtree(src)
        except Exception:
            pass

    # Final sanity
    need = [DEST / "config.json", DEST / "pytorch_model.bin"]
    if not all(p.exists() for p in need):
        raise SystemExit("Model files missing after flatten. Check ZIP structure.")


def main():
    print(f"[INFO] Target model dir: {DEST}")
    if DEST.exists() and any(DEST.iterdir()):
        print("[INFO] Model already present. Skipping download.")
        return

    if not MODEL_ZIP_URL:
        raise SystemExit("MODEL_ZIP_URL not set in .env")

    download_http(MODEL_ZIP_URL, ZIP_PATH)

    if ZIP_PATH.stat().st_size < 1024:
        raise SystemExit("Downloaded file is suspiciously small (likely wrong URL).")

    if MODEL_ZIP_SHA256:
        print("[INFO] Verifying SHA-256 …")
        digest = sha256_file(ZIP_PATH)
        if digest.lower() != MODEL_ZIP_SHA256.lower():
            raise SystemExit(f"SHA mismatch. Expected {MODEL_ZIP_SHA256}, got {digest}")

    print("[INFO] Unzipping …")
    unzip_and_flatten(ZIP_PATH)

    try: ZIP_PATH.unlink()
    except Exception: pass

    print(f"[INFO] Model ready at: {DEST}")

if __name__ == "__main__":
    main()
