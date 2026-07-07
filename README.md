# Document Q&A RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about any PDF or plain-text document. Built as Mini Project 2 of the Avirat.ai Agentic AI Engineering internship track.

## Architecture

[PDF/TXT file]
↓
load_document()          → extracts text per page (pypdf for PDFs, direct read for text)
↓
chunking strategy         → recursive_chunk() [default] or fixed_size_chunk()
↓
sentence-transformers     → embeds each chunk (all-MiniLM-L6-v2, local, no API key)
↓
ChromaDB                  → persists chunks + embeddings + metadata (source, page, chunk_id)
↓
[Query arrives]
↓
embed question → retrieve top-k chunks (vector similarity search)
↓
build_prompt()            → assembles retrieved chunks into context
↓
OpenRouter (poolside/laguna-xs-2.1:free) → generates answer
↓
{answer, sources} returned to client (JSON, or streamed via SSE)

### Components

| File | Responsibility |
|---|---|
| `app/config.py` | Central settings — chunk size, model names, API keys |
| `app/chunking.py` | Two chunking strategies: fixed-size and recursive |
| `app/ingest.py` | Load file → chunk → embed → store in ChromaDB |
| `app/rag.py` | Retrieve chunks → build prompt → call LLM (blocking and streaming) |
| `app/main.py` | FastAPI app: `/ingest`, `/ask`, `/ask/stream` |
| `evaluate.py` | Runs 5 known-answer questions, reports hit rate, compares chunking strategies |

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:
OPENROUTER_API_KEY=your-key-here

## Usage

**Ingest a document:**
```powershell
python -c "from app.ingest import ingest_document; ingest_document('sample_docs/yourfile.pdf', doc_id='my_doc')"
```

**Run the API:**
```powershell
uvicorn app.main:app --reload
```

**Ask a question:**
```powershell
python -c "
import requests
r = requests.post('http://127.0.0.1:8000/ask', json={'doc_id': 'my_doc', 'question': 'Your question here'})
print(r.json())
"
```

**Stream a question (SSE):**
```powershell
python -c "
import requests
with requests.post('http://127.0.0.1:8000/ask/stream', json={'doc_id': 'my_doc', 'question': 'Your question here'}, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode())
"
```

**Run the evaluation:**
```powershell
python evaluate.py
```

## Chunking Strategy Rationale

Two strategies are implemented:

- **Fixed-size chunking** (`fixed_size_chunk`): slices text into equal character windows with overlap. Simple and predictable, but can cut a sentence or fact in half at an arbitrary boundary.
- **Recursive chunking** (`recursive_chunk`, default): splits on paragraph breaks first, greedily merging pieces up to the chunk size limit. If a single paragraph is still too large, it recursively falls back to sentence-level, then word-level splitting — and to fixed-size slicing only as a last resort, so no text is ever lost.

**Recursive chunking is used as the default** (`CHUNKING_STRATEGY` in `config.py`) because it respects natural document structure and degrades gracefully — it only falls back to a harder split when a piece genuinely can't be divided any other way.

### Evaluation comparison

Both strategies were tested against the same 5-question evaluation set on the same source document:

| Strategy | Hit rate |
|---|---|
| Recursive (default) | 5/5 (100%) |
| Fixed-size | 5/5 (100%) |

Both strategies achieved a perfect hit rate on this document. This is likely because the source PDF is well-structured with short paragraphs and bullet points — most chunks fell well under the size limit regardless of strategy, so fixed-size chunking rarely had the opportunity to cut a fact awkwardly mid-sentence. On longer, prose-heavy, less structured documents, recursive chunking would be expected to show a clearer advantage, since it actively avoids splitting inside a sentence while fixed-size chunking does not check for this at all.

## Evaluation Report

5 questions with known correct answers were run against the ingested document. Full results and reasoning are printed by `evaluate.py`; summary:

- **Hit rate: 5/5 (100%)** on recursive chunking
- **Hit rate: 5/5 (100%)** on fixed-size chunking
- All answers correctly cited source page numbers and chunk IDs
- No hallucinated answers observed in testing

## Known Limitations

- Embedding model (`all-MiniLM-L6-v2`) runs locally via `sentence-transformers`; larger documents will take longer to embed on lower-end hardware.
- LLM calls go through OpenRouter's free tier (`poolside/laguna-xs-2.1:free`), which may have tighter rate limits than paid tiers.
- PDF text extraction relies on `pypdf`'s `.extract_text()`, which does not perform OCR — scanned/image-only PDFs will yield empty pages that are silently skipped.