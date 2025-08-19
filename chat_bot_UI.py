import streamlit as st
import requests
import uuid
import json
import re  # NEW

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
                #st.sidebar.code(json.dumps(res.json(), ensure_ascii=False, indent=2))
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
                #st.sidebar.code(json.dumps(res.json(), ensure_ascii=False, indent=2))
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
        return {"error": f"Error {r.status_code}: {r.text}"}
    try:
        return r.json()
    except Exception:
        return {"error": "Non-JSON response from server.", "raw": r.text}

# ---------- NEW: helpers to extract name, score, reason ----------
def extract_display_name(score_txt: str, snippet_txt: str) -> str:  # NEW
    name = None
    # Try JSON: {"name": ..., "file": ..., "filename": ...}
    try:
        obj = json.loads(score_txt or "")
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("file") or obj.get("filename")
    except Exception:
        pass

    # Fallback to "CV: ..." line inside snippet
    if not name:
        for line in (snippet_txt or "").splitlines():
            if line.strip().lower().startswith("cv:"):
                name = line.split(":", 1)[-1].strip()
                break

    # Normalize for display
    if name:
        name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        name = re.sub(r"\.(pdf|docx?)$", "", name, flags=re.I)
        name = re.sub(r"[_\s]+", " ", name).strip()
    return name or "Unknown Candidate"

def parse_score_and_reason(score_txt: str):  # NEW
    """
    Supports either:
      - Plain text: 'Score: 83/100\\nReason: ...'
      - JSON: {"score": 83, "explanation": "..."} (or "reason")
    Returns (score_val: float|None, reason: str|None)
    """
    score_val, reason = None, None

    # JSON path
    try:
        obj = json.loads(score_txt or "")
        if isinstance(obj, dict):
            if "score" in obj and obj["score"] is not None:
                score_val = float(obj["score"])
            reason = obj.get("explanation") or obj.get("reason")
    except Exception:
        pass

    # Plain-text fallbacks
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
                # ---- DEDUPE: group by name, keep highest score, then limit to k ----
                # UPDATED: store parsed reason too
                buckets = {}  # name -> {"item": best_item, "score_val": float|None, "reason": str|None, "all": [items]}
                for item in data:
                    score_txt = (item.get("score") or "").strip()
                    snippet_txt = (item.get("cv_snippet") or "")

                    name = extract_display_name(score_txt, snippet_txt)  # UPDATED
                    score_val, reason_val = parse_score_and_reason(score_txt)  # UPDATED

                    entry = buckets.get(name)
                    if not entry:
                        buckets[name] = {"item": item, "score_val": score_val, "reason": reason_val, "all": [item]}
                    else:
                        entry["all"].append(item)
                        cur = entry["score_val"]
                        if (score_val is not None) and (cur is None or score_val > cur):
                            entry["item"], entry["score_val"], entry["reason"] = item, score_val, reason_val

                # sort by score desc (None last), then name; apply k AFTER dedupe
                unique = sorted(
                    buckets.items(),
                    key=lambda kv: (-(kv[1]["score_val"] if kv[1]["score_val"] is not None else float("-inf")), kv[0].lower()),
                )[: int(k)]

                # Render + build transcr
