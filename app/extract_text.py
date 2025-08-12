from pathlib import Path
from typing import Union
from pypdf import PdfReader
from docx import Document
import re

def _clean(txt: str) -> str:
    # normalize whitespace
    return re.sub(r"\s+", " ", (txt or "")).strip()

def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    reader = PdfReader(str(file_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return _clean(" ".join(parts))

def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    doc = Document(str(file_path))
    parts = [p.text for p in doc.paragraphs]
    return _clean("\n".join(parts))

def extract_text(file_path: Union[str, Path], content_type: str | None = None) -> str:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf" or (content_type and "pdf" in content_type):
        return extract_text_from_pdf(file_path)
    if suffix == ".docx" or (content_type and "wordprocessingml" in content_type):
        return extract_text_from_docx(file_path)

    # fallback: try PDF then DOCX (best-effort)
    try:
        return extract_text_from_pdf(file_path)
    except Exception:
        return extract_text_from_docx(file_path)
