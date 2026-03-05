from __future__ import annotations

from openai import OpenAI


def rewrite_query_with_ollama(
    query: str,
    *,
    model: str = "llama3.2:latest",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must be non-empty")

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert technical resume writer. Write a single, highly technical "
                    "hypothetical resume bullet point that proves a candidate has the skills and include backend, frontend, and devops experience that the user is asking about in their query. "
                    "requested in the user's query. Use action verbs and technical terms. "
                    "Return ONLY the bullet point text."
                ),
            },
            {"role": "user", "content": cleaned},
        ],
    )

    rewritten = (response.choices[0].message.content or "").strip()
    if not rewritten:
        raise RuntimeError("Ollama returned an empty rewritten query")
    return rewritten
