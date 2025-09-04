# Healthcare LLM Starter (RAG + Safety + FastAPI)

This is a minimal, **safe-first** starter you can run on a laptop. It uses:
- **Ollama** to run a small local chat model (Qwen2.5 7B Instruct or Llama 3 8B Instruct).
- **RAG** over your trusted PDFs (WHO, CDC, NICE, hospital protocols) with **Chroma** vector DB.
- **Safety**: refusal on emergencies and low-confidence answers; always cites sources.

> Educational use only. Not a substitute for professional diagnosis or treatment.

---

## Quickstart (10 steps)

0) **Install Ollama** (https://ollama.com/download). Then pull a model:
```bash
ollama pull qwen2.5:7b-instruct
# or
ollama pull llama3:8b-instruct
```

1) **Create venv** (Python 3.10+), then:
```bash
pip install -r requirements.txt
```

2) **Put guideline PDFs** into: `data/guidelines/` (trusted sources only).

3) **Run the API**:
```bash
uvicorn app.main:app --reload --port 8000
```

4) **Ask a question**:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json"   -d '{"question": "What are first-line treatments for hypertension in adults?"}'
```

5) You’ll get `answer`, `confidence`, and `sources` (file + page).

6) **Low confidence?** The system abstains and recommends seeing a clinician.

7) **Add more PDFs** and restart to expand knowledge.

8) **(Optional) Fine-tune** with QLoRA via `scripts/train_qlora.py` (see file).

9) **(Optional) Simple UI**:
```bash
streamlit run scripts/streamlit_app.py
```

---

## Files
- `app/main.py` — FastAPI server with `/ask` endpoint.
- `app/rag.py` — RAG pipeline (ingestion, retrieval, prompting).
- `app/safety.py` — safety/abstention checks.
- `app/config.py` — model & thresholds.
- `scripts/train_qlora.py` — minimal QLoRA SFT recipe.
- `scripts/streamlit_app.py` — tiny Streamlit client.
- `data/guidelines/` — drop PDFs here.
- `storage/` — vector DB (auto-created).

Use responsibly. Stay safe!
