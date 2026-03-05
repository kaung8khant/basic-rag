from __future__ import annotations

from typing import Any

from openai import OpenAI


def build_context_from_matches(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return ""

    lines: list[str] = []
    for item in matches:
        chunk = item.get("chunk", {})
        lines.append(
            f"[k={item.get('k')}, faiss_id={item.get('faiss_id')}, score={item.get('score')}]"
        )
        lines.append(str(chunk.get("text", "")))
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_ollama(
    question: str,
    *,
    context: str,
    model: str = "llama3.2:latest",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be non-empty")

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are reviewing a candidate's resume and answering a question about their experience based on the provided context. "
                    "Answer the question using only the provided context. "
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{cleaned}\n\nContext:\n{context}",
            },
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    return answer
