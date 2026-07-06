import os
from dotenv import load_dotenv

load_dotenv()

#Storage configuration
CHROMA_PERSIST_DIR = "./chroma_db"  #tells ChromaDB where to store the embeddings and metadata
COLLECTION_NAME = "documents"

#Embeddings
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

#Chunking
CHUNK_SIZE = 800 #chars per chunk
OVERLAP_SIZE = 150 #overlap between consecutive chunks

#Retrieval
TOP_K = 4 #how many chunks to retrive per question

#LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "poolside/laguna-xs-2.1:free"
