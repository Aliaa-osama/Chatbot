import streamlit as st
import requests
import uuid
import json

# ---------- Config ----------
st.set_page_config(page_title="CV RAG Chat", layout="centered")
st.title("CV RAG Chat — Aliaa Osama Alkady")

# Where your FastAPI is running
API_BASE = "https://chatbot-43d0.onrender.com/"

# ---------- Sidebar ----------
k = st.sidebar.slider("Top Candidates (k)", 1, 10, 2)

st.sidebar.divider()
st.sidebar.subheader("Upload CV(s)")

# Single upload
single_file = st.sidebar.file_uploader("Upload one PDF", type=["pdf"], key="single_pdf")
if st.sidebar.button("Upload"):
    if single_file is None:
        st.sidebar.warning("Choose a PDF first.")
    else:
        try:
            res = requests.post(
                f"{API_BASE}/upload",
                files={"file": (single_file.name, single_file.getvalue(), "application/pdf")},
                timeout=180,
            )
            if res.ok:
                st.sidebar.success("Uploaded & indexed ✅")
                st.sidebar.code(json.dumps(res.json(), ensure_ascii=False, indent=2))
            else:
                st.sidebar.error(f"Upload failed: {res.status_code}")
                st.sidebar.code(res.text)
        except requests.RequestException as e:
            st.sidebar.error(f"Upload error: {e}")

# Batch upload
batch_files = st.sidebar.file_uploader("Upload multiple PDFs", type=["pdf"], accept_multiple_files=True, key="batch_pdf")
if st.sidebar.button("Upload batch"):
    if not batch_files:
        st.sidebar.warning("Choose one or more PDFs first.")
    else:
        try:
            payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in batch_files]
            res = requests.post(f"{API_BASE}/upload-batch", files=payload, timeout=600)
            if res.ok:
                st.sidebar.success("Batch uploaded & indexed ✅")
                st.sidebar.code(json.dumps(res.json(), ensure_ascii=False, indent=2))
            else:
                st.sidebar.error(f"Batch failed: {res.status_code}")
                st.sidebar.code(res.text)
        except requests.RequestException as e:
            st.sidebar.error(f"Batch error: {e}")

st.sidebar.divider()
if st.sidebar.button("Clear chat"):
    st.session_state.pop("messages", None)
    st.rerun()

# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

def pretty(x):
    if isinstance(x, list):
        return "\n\n---\n\n".join(x) if x else "No results."
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)

def post_json(path, payload, timeout=180):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    if not r.ok:
        return f"Error {r.status_code}: {r.text}"
    try:
        return r.json()
    except Exception:
        return r.text

# ---------- Render history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Type your job description.")
if prompt:
    # 1) show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) call /ask (retrieve top-k snippets)
    payload = {"query": prompt, "k": int(k)}
    endpoint = "/ask"

    # 3) show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                data = post_json(endpoint, payload)
                bot_response = pretty(data)
            except requests.RequestException as e:
                bot_response = f"Error: {e}"
            st.markdown(f"```\n{bot_response}\n```")

    # 4) store assistant message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
