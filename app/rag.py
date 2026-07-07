'''
Retrieval + LLM generation.
Handles spec items #10 and #11:
  - embed the question → retrieve top-k chunks → feed to LLM with context
  - expose as {question, doc_id} -> {answer, sources}
'''

from openai import OpenAI

from app.config import (
    TOP_K,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
)
from app.ingest import get_embedding_model, get_chroma_collection


def retrieve_chunks(question: str, doc_id: str = None, top_k: int = TOP_K) -> list[dict]:
    """
    Embeds the question, queries ChromaDB, and returns the top-k matching
    chunks along with their metadata.
    """
    model = get_embedding_model()
    collection = get_chroma_collection()

    query_embedding = model.encode([question]).tolist()

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
    )

    # results come back as parallel lists nested one level (one list per query embedding)
    # since we only sent one question, we only care about index [0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []
    for doc_text, meta in zip(documents, metadatas):
        chunks.append({
            "text": doc_text,
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"],
        })

    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Combines retrieved chunks into a context block and builds the final prompt.
    """
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, doc_id: str = None, top_k: int = TOP_K) -> dict:
    """
    Full RAG flow: retrieve chunks, build prompt, call the LLM, return answer + sources.
    """
    chunks = retrieve_chunks(question, doc_id=doc_id, top_k=top_k)

    if not chunks:
        return {"answer": "No relevant context found for this question.", "sources": []}

    prompt = build_prompt(question, chunks)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content

    sources = [
        {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}