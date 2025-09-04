# scripts/force_ingest.py
from app.rag import RAGPipeline

p = RAGPipeline()
p.ingest()
print("Ingest finished.")
