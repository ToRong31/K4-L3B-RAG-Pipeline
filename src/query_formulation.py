"""One OpenAI request produces Vietnamese and English retrieval queries."""

import json
import os

from dotenv import load_dotenv

load_dotenv()


def formulate_query(query: str) -> tuple[str, str]:
    """Return (Vietnamese query, English query); preserve input on API failure."""
    original = query.strip()
    if not original:
        return "", ""
    if os.getenv("QUERY_FORMULATION_ENABLED", "false").lower() != "true":
        return original, ""
    if not os.getenv("OPENAI_API_KEY"):
        return original, ""
    try:
        from openai import OpenAI

        client = OpenAI(timeout=15.0, max_retries=1)
        response = client.responses.create(
            model=os.getenv("QUERY_FORMULATION_MODEL", "gpt-4.1-mini"),
            instructions=(
                "Rewrite the user's question for document retrieval. Return one precise "
                "Vietnamese query and its English translation. Preserve names, numbers, "
                "document codes, and intent. Do not answer the question."
            ),
            input=original,
            text={"format": {
                "type": "json_schema",
                "name": "retrieval_queries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query_vi": {"type": "string"},
                        "query_en": {"type": "string"},
                    },
                    "required": ["query_vi", "query_en"],
                    "additionalProperties": False,
                },
            }},
        )
        result = json.loads(response.output_text)
        return result["query_vi"].strip() or original, result["query_en"].strip()
    except Exception:
        return original, ""
