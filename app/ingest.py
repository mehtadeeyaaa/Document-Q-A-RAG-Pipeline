'''
Load a PDF or text file (accept any PDF or plain-text file — spec item #7)
Run it through one of your two chunking functions
Embed each chunk with sentence-transformers
Store everything in ChromaDB with metadata (source filename, page number, chunk ID — spec item #9)
'''

from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDINGS_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    OVERLAP_SIZE,
)
from app.chunking import fixed_size_chunk, recursive_chunk


def load_text(file_path: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
    return [{"page": 1, "text": text}]


def load_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def load_document(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    if ext == ".pdf":
        pages = load_pdf(file_path)
    elif ext in (".txt", ".md"):
        pages = load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {"filename": filename, "pages": pages}


_model = None  # cache so we don't reload the embedding model every call

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDINGS_MODEL)
    return _model


def get_chroma_collection(collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def ingest_document(file_path: str, doc_id: str, chunking_strategy: str = "recursive",
                     collection_name: str = COLLECTION_NAME) -> int:
    """
    Loads a file, chunks it, embeds each chunk, and stores it in ChromaDB.
    Returns the number of chunks stored.
    """
    doc = load_document(file_path)
    model = get_embedding_model()
    collection = get_chroma_collection(collection_name)

    chunk_fn = fixed_size_chunk if chunking_strategy == "fixed" else recursive_chunk

    all_texts = []
    all_ids = []
    all_metadatas = []

    for page in doc["pages"]:
        page_num = page["page"]
        page_chunks = chunk_fn(page["text"], CHUNK_SIZE, OVERLAP_SIZE)

        for chunk_index, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                continue

            chunk_id = f"{doc_id}_p{page_num}_c{chunk_index}"
            all_texts.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": doc["filename"],
                "page": page_num,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    if not all_texts:
        return 0

    embeddings = model.encode(all_texts).tolist()

    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    return len(all_texts)