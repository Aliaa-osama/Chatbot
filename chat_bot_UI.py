import streamlit as st
import requests
import uuid
import json

# ---------- Config ----------
st.set_page_config(page_title="CV RAG Chat", layout="centered")
st.title("CV RAG Chat — Aliaa Osama Alkady")

# Where your FastAPI is running (no trailing slash)
API_BASE = "https://chatbot-43d0.onrender.com"

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

def post_json(path, payload, timeout=180):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    if not r.ok:
        # Return a dict error so downstream rendering is consistent
        return {"error": f"Error {r.status_code}: {r.text}"}
    try:
        return r.json()
    except Exception:
        return {"error": "Non-JSON response from server.", "raw": r.text}

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

    # 2) call /ask (retrieve + score top-k snippets)
    payload = {"query": prompt, "k": int(k)}
    endpoint = "/ask"

    # 3) show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = post_json(endpoint, payload)

            # Handle explicit error dict
            if isinstance(data, dict) and "error" in data:
                bot_response = data["error"]
                st.error(bot_response)

            # Handle new format: list of dicts {cv_snippet, score}
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # Build a human-friendly transcript message to store in history
                combined_lines = []
                for i, item in enumerate(data, start=1):
                    score_txt = item.get("score", "").strip()
                    snippet = item.get("cv_snippet", "").strip()
                    combined_lines.append(f"**Candidate {i}**\n\n{score_txt}\n\n```\n{snippet}\n```")

                    # Visual card per candidate
                    with st.container(border=True):
                        st.markdown(f"### Candidate {i}")
                        if score_txt:
                            st.markdown(f"**{score_txt.splitlines()[0]}**")  # "Score: x/100"
                            if len(score_txt.splitlines()) > 1:
                                st.markdown(score_txt.splitlines()[1])      # "Reason: ..."
                        if snippet:
                            with st.expander("View retrieved snippet"):
                                st.code(snippet)
                bot_response = "\n\n---\n\n".join(combined_lines) if combined_lines else "No results."

            # Handle old format: list of strings
            elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                bot_response = "\n\n---\n\n".join(data) if data else "No results."
                st.markdown(f"```\n{bot_response}\n```")

            else:
                # Fallback: show raw JSON
                bot_response = json.dumps(data, ensure_ascii=False, indent=2)
                st.code(bot_response)

    # 4) store assistant message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
