from pathlib import Path

# ---- Model runtime (Ollama must be running locally) ----
OLLAMA_MODEL = "qwen2.5:7b-instruct"  # or "llama3:8b-instruct"

# ---- Embeddings ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- RAG settings ----
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "guidelines"
DB_DIR = Path(__file__).resolve().parent.parent / "storage"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
TOP_K = 4


TEMPERATURE = 0.0 
# Confidence threshold (0..1). Below this → abstain.
CONF_THRESHOLD = 0.45

# Max characters from retrieved context to keep prompt compact
MAX_CONTEXT_CHARS = 8000
