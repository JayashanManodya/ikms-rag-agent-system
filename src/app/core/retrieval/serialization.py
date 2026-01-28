"""Utilities for serializing document chunks into strings."""

from typing import List
from langchain_core.documents import Document

def serialize_chunks(docs: List[Document]) -> str:
    """Consolidate document chunks into a single formatted string.

    Args:
        docs: List of Document objects.

    Returns:
        Formatted CONTEXT string.
    """
    if not docs:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.replace("\n", " ").strip()
        context_parts.append(f"Chunk {i} (page={page}): {content}")

    return "\n\n".join(context_parts)
