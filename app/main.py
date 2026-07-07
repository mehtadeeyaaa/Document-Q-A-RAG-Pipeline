from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import generate_answer
from app.ingest import ingest_document

app = FastAPI(title="Document Question Answering API")

class AskRequest(BaseModel):
    doc_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = generate_answer(request.question, request.doc_id)
    return result

class IngestRequest(BaseModel):
    doc_id: str
    file_path: str
    chunking_strategy: str = "recursive"

@app.post("/ingest")
def ingest(request: IngestRequest):
    count = ingest_document(request.file_path, request.doc_id, request.chunking_strategy)
    return {"chunks_stored": count}