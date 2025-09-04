from fastapi import FastAPI
from pydantic import BaseModel
from .rag import RAGPipeline
from .safety import check_emergency, DISCLAIMER, check_model_output
from .config import CONF_THRESHOLD

app = FastAPI(title="Healthcare LLM (Safe RAG)")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list

# Initialize RAG pipeline
pipeline = RAGPipeline()
pipeline.load_or_create()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 🚨 Step 1: Emergency check (before generation)
    emerg = check_emergency(req.question)
    if emerg:
        return AskResponse(answer=emerg + " " + DISCLAIMER, confidence=0.0, sources=[])

    # 🤖 Step 2: Get model output
    result = pipeline.ask(req.question)

    # ⚖️ Step 3A: Confidence check (low → abstain)
    if result["confidence"] < CONF_THRESHOLD:
        msg = (
            "I can't provide medical advice here based on the available sources. "
            "Please see a clinician. " + DISCLAIMER
        )
        return AskResponse(
            answer=msg, confidence=result["confidence"], sources=result["sources"]
        )

    # 🔒 Step 3B: Back-gate check on model output
    model_check = check_model_output(result["answer"])
    if model_check:
        return AskResponse(
            answer=model_check + "\n\n" + DISCLAIMER,
            confidence=0.0,
            sources=result["sources"],
        )

    # ✅ Step 4: Safe response with disclaimer
    result["answer"] = result["answer"].strip() + f"\n\n—\n{DISCLAIMER}"
    return AskResponse(**result)
