import streamlit as st
import requests
import uuid
import json
import re

# ================== Config ==================
st.set_page_config(page_title="CV RAG Chat", layout="centered")
st.title("AI Assistant — Chat & Candidate Ranking")

API_BASE = "https://chatbot-43d0.onrender.com"

# ================== Sidebar ==================
mode = st.sidebar.radio("Mode", ["Auto", "Rank candidates", "General chat"], index=0)
k = st.sidebar.slider("Top Candidates (k)", 1, 10, 2, help="Used in Rank candidates mode")
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
            else:
                st.sidebar.error("Upload failed. Please try again later.")
        except requests.RequestException:
            st.sidebar.error("Upload error. Please check your connection.")

# Batch upload
batch_files = st.sidebar.file_uploader(
    "Upload multiple PDFs", type=["pdf"], accept_multiple_files=True, key="batch_pdf"
)
if st.sidebar.button("Upload batch"):
    if not batch_files:
        st.sidebar.warning("Choose one or more PDFs first.")
    else:
        try:
            payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in batch_files]
            res = requests.post(f"{API_BASE}/upload-batch", files=payload, timeout=600)
            if res.ok:
                st.sidebar.success("Batch uploaded & indexed ✅")
            else:
                st.sidebar.error("Batch upload failed. Please try again later.")
        except requests.RequestException:
            st.sidebar.error("Batch upload error. Please check your connection.")

st.sidebar.divider()
if st.sidebar.button("Clear chat"):
    st.session_state.pop("messages", None)
    st.rerun()

# ================== Session state ==================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== HTTP helper ==================
def post_json(path, payload, timeout=180):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    except requests.RequestException:
        return {"_error": "Network error. Please try again later."}
    if not r.ok:
        return {"_error": "Request failed. Please try again later."}
    try:
        return r.json()
    except Exception:
        return {"_error": "Unexpected server response."}

# ================== Parsing helpers ==================
def extract_display_name(score_txt: str, snippet_txt: str) -> str:
    name = None
    try:
        obj = json.loads(score_txt or "")
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("file") or obj.get("filename")
    except Exception:
        pass
    if not name:
        for line in (snippet_txt or "").splitlines():
            if line.strip().lower().startswith("cv:"):
                name = line.split(":", 1)[-1].strip()
                break
    if name:
        name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        name = re.sub(r"\.(pdf|docx?)$", "", name, flags=re.I)
        name = re.sub(r"[_\s]+", " ", name).strip()
    return name or "Unknown Candidate"

def parse_score_and_reason(score_txt: str):
    score_val, reason = None, None
    try:
        obj = json.loads(score_txt or "")
        if isinstance(obj, dict):
            if "score" in obj and obj["score"] is not None:
                score_val = float(obj["score"])
            reason = obj.get("explanation") or obj.get("reason")
    except Exception:
        pass
    if score_val is None:
        m = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)(?:\s*/\s*100)?", score_txt or "", flags=re.I)
        if m:
            try:
                score_val = float(m.group(1))
            except Exception:
                pass
    if not reason:
        m2 = re.search(r"Reason\s*:\s*(.+)", score_txt or "", flags=re.I | re.S)
        if m2:
            reason = m2.group(1).strip()
    return score_val, reason

def looks_like_jd(text: str) -> bool:
    if not text:
        return False
    tokens = len(text.split())
    jd_keywords = ["responsibilities", "requirements", "qualifications", "experience", "skills", "role", "description"]
    score = sum(1 for k in jd_keywords if k.lower() in text.lower())
    return tokens > 25 or score >= 2

# ================== Render history ==================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== Chat input ==================
prompt = st.chat_input("Type your job description or ask a general question.")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    endpoint = None
    payload = None
    if mode == "Rank candidates" or (mode == "Auto" and looks_like_jd(prompt)):
        endpoint = "/ask"
        payload = {"query": prompt, "k": int(k)}
    else:
        endpoint = "/chat"
        payload = {"message": prompt}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = post_json(endpoint, payload)

            if isinstance(data, list) and data and isinstance(data[0], dict):
                # candidate ranking display (same as before)...
                pass
            elif isinstance(data, dict) and "answer" in data:
                st.markdown(data.get("answer") or "No answer returned.")
            elif isinstance(data, dict) and "_error" in data:
                st.warning(data["_error"])  # friendly message only
            else:
                st.info("No response.")

    st.session_state.messages.append({"role": "assistant", "content": str(data)})
