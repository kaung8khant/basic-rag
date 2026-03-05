from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


def extract_pdf_segments(content: bytes) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    with fitz.open(stream=content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue
            segments.append({"text": page_text, "page": page_num, "section": None})

    return segments


def extract_markdown_segments(content: bytes) -> list[dict[str, Any]]:
    raw = content.decode("utf-8", errors="replace")
    lines = raw.splitlines()
    segments: list[dict[str, Any]] = []
    current_section = "document"
    buffer: list[str] = []

    def flush_buffer() -> None:
        text = "\n".join(buffer).strip()
        if text:
            segments.append({"text": text, "page": None, "section": current_section})

    for line in lines:
        if line.lstrip().startswith("#"):
            flush_buffer()
            buffer.clear()
            current_section = line.lstrip("#").strip() or "untitled-section"
            continue
        buffer.append(line)

    flush_buffer()
    return segments


def extract_segments(content: bytes, ext: str) -> list[dict[str, Any]]:
    if ext == ".pdf":
        return extract_pdf_segments(content)
    if ext in {".md", ".markdown"}:
        return extract_markdown_segments(content)
    raise ValueError(f"Unsupported extension: {ext}")


def extract_and_enrich_segments(
    content: bytes,
    ext: str,
    saved_as: str,
    owner_id: str,
    visibility: str = "private",
) -> list[dict[str, Any]]:
    segments = extract_segments(content, ext)
    created_at = datetime.now().isoformat()
    doc_id = Path(saved_as).stem

    for segment in segments:
        segment["doc_id"] = doc_id
        segment["owner_id"] = owner_id
        segment["created_at"] = created_at
        segment["visibility"] = visibility

    return segments
