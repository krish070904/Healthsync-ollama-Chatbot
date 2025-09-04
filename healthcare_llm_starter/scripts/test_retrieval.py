# scripts/test_retrieval.py
from app.rag import RAGPipeline

q = "What are the common symptoms of diabetes?"
p = RAGPipeline()
p.load_or_create()
docs = p.retrieve(q)
print("Retrieved:", len(docs))
for i,d in enumerate(docs, start=1):
    print(i, d.source, d.page, d.score)
    print(d.snippet[:400])
    print("-"*40)
