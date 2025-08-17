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
                # ---- DEDUPE INLINE: group by name, keep highest score, then limit to k ----
                buckets = {}  # name -> {"item": best_item, "score_val": float|None, "all": [items]}
                for item in data:
                    score_txt = (item.get("score") or "").strip()
                    snippet_txt = (item.get("cv_snippet") or "")

                    # derive display name (try JSON->name/file/filename, else "CV: ..." line)
                    name = None
                    try:
                        obj = json.loads(score_txt)
                        if isinstance(obj, dict):
                            name = obj.get("name") or obj.get("file") or obj.get("filename")
                    except Exception:
                        pass
                    if not name:
                        for line in snippet_txt.splitlines():
                            if line.strip().lower().startswith("cv:"):
                                name = line.split(":", 1)[-1].strip()
                                break
                    if name:
                        name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        name = re.sub(r"\.(pdf|docx?)$", "", name, flags=re.I).replace("_", " ").strip()
                    if not name:
                        name = "Unknown Candidate"

                    # numeric score (JSON 'score' or "Score: 78[/100]")
                    score_val = None
                    try:
                        obj = json.loads(score_txt)
                        if isinstance(obj, dict) and "score" in obj:
                            score_val = float(obj["score"])
                    except Exception:
                        pass
                    if score_val is None:
                        m = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)(?:\s*/\s*100)?", score_txt, flags=re.I)
                        if m:
                            try:
                                score_val = float(m.group(1))
                            except Exception:
                                pass

                    entry = buckets.get(name)
                    if not entry:
                        buckets[name] = {"item": item, "score_val": score_val, "all": [item]}
                    else:
                        entry["all"].append(item)
                        cur = entry["score_val"]
                        if (score_val is not None) and (cur is None or score_val > cur):
                            entry["item"], entry["score_val"] = item, score_val

                # sort by score desc (None last), then name; apply k AFTER dedupe
                unique = sorted(
                    buckets.items(),
                    key=lambda kv: (-(kv[1]["score_val"] if kv[1]["score_val"] is not None else float("-inf")), kv[0].lower()),
                )[: int(k)]

                # Render + build transcript
                combined_lines = []
                for idx, (name, pack) in enumerate(unique, start=1):
                    best_item = pack["item"]
                    score_txt = (best_item.get("score") or "").strip()
                    snippet = (best_item.get("cv_snippet") or "").strip()

                    with st.container(border=True):
                        st.markdown(f"### {name if name != 'Unknown Candidate' else f'Candidate {idx}'}")

                        # Friendly score summary if possible
                        shown_score = None
                        try:
                            obj = json.loads(score_txt)
                            if isinstance(obj, dict):
                                lines = []
                                if "score" in obj:
                                    lines.append(f"**Score:** {obj['score']}/100")
                                if "explanation" in obj and obj["explanation"]:
                                    lines.append(f"**Reason:** {obj['explanation']}")
                                shown_score = "\n\n".join(lines) if lines else None
                        except Exception:
                            pass
                        if not shown_score and score_txt:
                            shown_score = score_txt.splitlines()[0]
                        if shown_score:
                            st.markdown(shown_score)

                        if snippet:
                            with st.expander("View best retrieved snippet"):
                                st.code(snippet)

                        others = [it for it in pack["all"] if it is not best_item]
                        if others:
                            with st.expander(f"Other retrieved snippets ({len(others)})"):
                                for j, it in enumerate(others, start=1):
                                    st.markdown(f"**Snippet {j}**")
                                    st.code((it.get("cv_snippet") or "").strip())

                    # Chat history block (compact)
                    pretty = name if name != "Unknown Candidate" else f"Candidate {idx}"
                    score_line = ""
                    try:
                        obj = json.loads(score_txt)
                        if isinstance(obj, dict) and "score" in obj:
                            score_line = f"Score: {obj['score']}/100"
                    except Exception:
                        if score_txt:
                            score_line = score_txt.splitlines()[0]
                    block = [f"**{pretty}**"]
                    if score_line:
                        block.append(score_line)
                    if snippet:
                        block += ["```", snippet, "```"]
                    combined_lines.append("\n\n".join(block))

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
