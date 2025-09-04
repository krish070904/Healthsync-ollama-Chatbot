import streamlit as st
import requests
st.set_page_config(page_title="Healthcare LLM (RAG)", page_icon="🩺")

st.title("🩺 Healthcare LLM (RAG) — Local & Safe")
st.caption("Educational use only. Not a substitute for professional care.")

api_url = st.text_input("API URL", value="http://localhost:8000/ask")
q = st.text_area("Ask a medical, guideline-backed question:", height=120)

if st.button("Ask"):
    with st.spinner("Thinking..."):
        r = requests.post(api_url, json={"question": q}, timeout=300)
        if r.ok:
            data = r.json()
            st.write("### Answer")
            st.write(data["answer"])
            st.write("### Confidence", data["confidence"])
            st.write("### Sources")
            for s in data.get("sources", []):
                st.write(f"- {s['source']} (page {s['page']}) — score {s['score']:.2f}")
        else:
            st.error(f"Error: {r.status_code}\n{r.text}")
