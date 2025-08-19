import streamlit as st
import requests
import uuid
import json
import re

# ================== Config ==================
st.set_page_config(page_title="CV RAG Chat", layout="centered")
st.title("CV RAG Chat — Aliaa Osama Alkady")

# Where your FastAPI is running (no trailing slash)
API_BASE = "https://chatbot-43d0.onrender.com"

# ================== Sidebar ==================
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
            else:
                st.sidebar.error(f"Batch failed: {res.status_code}")
                st.sidebar.code(res.text)
        except requests.RequestException as e:
            st.sidebar.error(f"Batch error: {e}")

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
    except requests.RequestException as e:
        return {"error": f"Network error: {e}"}
    if not r.ok:
        return {"error": f"Error {r.status_code}: {r.text}"}
    try:
        return r.json()
    except Exception:
        return {"error": "Non-JSON response from server.", "raw": r.text}

# ================== Parsing helpers ==================
def extract_display_name(score_txt: str, snippet_txt: str) -> str:
    """Prefer JSON name/file/filename; fallback to 'CV: ...' line; normalize for display."""
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
    """
    Accepts either:
      - Plain text: 'Score: 83/100\\nReason: ...'
      - JSON: {"score": 83, "explanation": "..."} (or "reason")
    Returns (score_val: float|None, reason: str|None)
    """
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

# ================== Render history ==================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== Chat input ==================
prompt = st.chat_input("Type your job description.")
if prompt:
    # 1) show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) call /ask
    payload = {"query": prompt, "k": int(k)}
    endpoint = "/ask"

    # 3) show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = post_json(endpoint, payload)

            # For debugging, uncomment:
            # st.write("RAW:", data)

            if isinstance(data, dict) and "error" in data:
                bot_response = data["error"]
                st.error(bot_response)

            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # ---- DEDUPE: group by name, keep highest score, then limit to k ----
                buckets = {}  # name -> {"item": best_item, "score_val": float|None, "reason": str|None, "all": [items]}
                for item in data:
                    score_txt = (item.get("score") or "").strip()
                    snippet_txt = (item.get("cv_snippet") or "")
                    name = extract_display_name(score_txt, snippet_txt)
                    score_val, reason_val = parse_score_and_reason(score_txt)

                    entry = buckets.get(name)
                    if not entry:
                        buckets[name] = {"item": item, "score_val": score_val, "reason": reason_val, "all": [item]}
                    else:
                        entry["all"].append(item)
                        cur = entry["score_val"]
                        if (score_val is not None) and (cur is None or score_val > cur):
                            entry["item"], entry["score_val"], entry["reason"] = item, score_val, reason_val

                # sort by score desc (None last), then by name; apply top-k AFTER dedupe
                unique = sorted(
                    buckets.items(),
                    key=lambda kv: (-(kv[1]["score_val"] if kv[1]["score_val"] is not None else float("-inf")), kv[0].lower()),
                )[: int(k)]

                # ---------- Render + build transcript ----------
                combined_lines = []
                for idx, (name, pack) in enumerate(unique, start=1):
                    best_item = pack["item"]
                    score_txt = (best_item.get("score") or "").strip()
                    snippet = (best_item.get("cv_snippet") or "").strip()

                    with st.container(border=True):
                        display_name = name if name != "Unknown Candidate" else f"Candidate {idx}"
                        st.markdown(f"### {display_name}")

                        parsed_score, parsed_reason = parse_score_and_reason(score_txt)
                        if parsed_score is not None:
                            st.markdown(f"**Score:** {int(parsed_score) if parsed_score.is_integer() else parsed_score}/100")
                        if parsed_reason:
                            st.markdown(f"**Reason:** {parsed_reason}")

                        if parsed_score is None and parsed_reason is None and score_txt:
                            st.markdown(score_txt.splitlines()[0])

                        if snippet:
                            with st.expander("View best retrieved snippet"):
                                st.code(snippet)

                        # other retrieved snippets (different names only)
                        others = []
                        for it in pack["all"]:
                            if it is best_item:
                                continue
                            other_name = extract_display_name((it.get("score") or ""), (it.get("cv_snippet") or ""))
                            if other_name != name:
                                others.append(it)
                        if others:
                            with st.expander(f"Other retrieved snippets ({len(others)})"):
                                for j, it in enumerate(others, start=1):
                                    st.markdown(f"**Snippet {j}**")
                                    st.code((it.get("cv_snippet") or "").strip())

                    # compact transcript block
                    pretty = display_name
                    score_line = ""
                    if parsed_score is not None:
                        score_line = f"Score: {int(parsed_score) if parsed_score.is_integer() else parsed_score}/100"
                    elif score_txt:
                        score_line = score_txt.splitlines()[0]

                    block = [f"**{pretty}**"]
                    if score_line:
                        block.append(score_line)
                    if parsed_reason:
                        block.append(f"Reason: {parsed_reason}")
                    if snippet:
                        block += ["```", snippet, "```"]
                    combined_lines.append("\n\n".join(block))

                bot_response = "\n\n---\n\n".join(combined_lines) if combined_lines else "No results."

            elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                # old format: list of strings
                bot_response = "\n\n---\n\n".join(data) if data else "No results."
                st.markdown(f"```\n{bot_response}\n```")

            else:
                # fallback: show raw JSON
                bot_response = json.dumps(data, ensure_ascii=False, indent=2)
                st.code(bot_response)

    # 4) store assistant message for history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
