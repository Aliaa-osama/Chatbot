# CV RAG Chat — FastAPI + Streamlit (LangChain, Gemini, Chroma)

A minimal Retrieval-Augmented system for **querying and scoring CVs (PDFs)** against a given **job description**.  
Back-end is **FastAPI** with **Chroma** vector search and **LangChain Google Generative AI** (Gemini).  
Front-end is a lightweight **Streamlit** chat UI that consumes the API.

**🚀 Deployments:**  
- **Backend (FastAPI on Render):** https://chatbot-43d0.onrender.com  
- **Frontend (Streamlit Cloud):** https://chatbot-b2mcvnqwe9xjcsv43egn9u.streamlit.app  

---

## ✨ Features

- Upload single or multiple **PDF CVs** and index them.
- Provide a **job description** → system retrieves **top-k relevant CV snippets** and **scores** each one.
- Score output:  
  ```
  Score: <number>/100
  Reason: <short explanation>
  ```
- Rebuild the index from a folder of PDFs.
- Simple health and docs endpoints.
- Ready-to-run Streamlit chat interface.

---

## 🧱 Tech Stack

- **Python 3.10+**
- **FastAPI** + **Uvicorn**
- **Streamlit**
- **LangChain** (`langchain_community`, `langchain_google_genai`)
- **Chroma** (vector store)
- **PyPDF** loaders
- **dotenv** for config

---

## 📁 Project Structure

```
.
├─ main.py                # FastAPI backend (upload, ask=retrieve+score, reindex)
├─ streamlit_app.py       # Streamlit chat UI
├─ .env                   # Environment variables (see below)
├─ requirements.txt       # Dependencies
└─ cvs/                   # Default folder where uploaded PDFs are stored
```

> In `main.py`, PDFs default to `./cvs`. The API will create this folder if missing.

---

## 🔐 Configuration

Create a `.env` in the project root:

```ini
google_api=YOUR_GOOGLE_GEMINI_API_KEY
CHROMA_COLLECTION=resumes   # optional; default = "resumes"
```

⚠️ **Never commit real keys.** Add `.env` to `.gitignore`.

---

## 📦 Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn python-dotenv streamlit requests
pip install langchain langchain-community langchain-google-genai
pip install chromadb pypdf
```

Uses **Google Generative AI Embeddings** (`models/embedding-001`) and **gemini-1.5-flash**.

---

## ▶️ Running Locally

**Backend (FastAPI)**  
```bash
python main.py
# or: uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- Swagger docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

**Frontend (Streamlit)**  
```bash
streamlit run streamlit_app.py
```

- Edit `API_BASE` in `streamlit_app.py`:
```python
API_BASE = "http://127.0.0.1:8000"   # local
# API_BASE = "https://chatbot-43d0.onrender.com"   # deployed
```

---

## 🔌 API Reference

### `GET /health`
Health check:
```json
{ "status": "ok" }
```

### `GET /`
Welcome message:
```json
{ "message": "Welcome to the Chatbot + Resume Scorer API! Use /docs for API documentation." }
```

### `POST /upload`
Upload one PDF CV and index it.

- **Request (multipart/form-data):** `file=@resume.pdf`  
- **Response:**
```json
{
  "status": "uploaded",
  "saved_as": "cvs/resume.pdf",
  "vector_count": 42
}
```

### `POST /upload-batch`
Upload multiple PDFs.  
Returns list of saved files + vector count.

### `POST /reindex`
Rebuild vector store from a directory.

- **Body:**
```json
{ "directory": "E:/NTI/chatbot/cvs" }
```
(If omitted, uses default `./cvs`.)

- **Response:**
```json
{
  "status": "reindexed",
  "directory": "E:/NTI/chatbot/cvs",
  "num_docs": 123
}
```

### `POST /ask`
Retrieve **top-k CV snippets** and **score** them.

- **Body:**
```json
{ "query": "Senior data scientist with NLP and MLOps", "k": 2 }
```

- **Response (list of objects):**
```json
[
  {
    "cv_snippet": "CV: john_doe.pdf\nSimilarityDistance: 0.1234\nContent: ...",
    "score": "Score: 86/100\nReason: Strong ML, Python; lacks NLP production."
  },
  {
    "cv_snippet": "CV: jane_smith.pdf\nSimilarityDistance: 0.2050\nContent: ...",
    "score": "Score: 72/100\nReason: Good analytics; limited deep learning."
  }
]
```

---

## 🖥️ Streamlit UI

- Sidebar supports single/batch uploads.  
- Chat box sends text to `/ask` with slider-controlled `k`.  
- Results show **Score + Reason** and allow expanding to see retrieved snippet.  

> Ensure `API_BASE` has **no trailing slash** (use `...com` not `...com/`) to avoid `//ask`.

Live UI: https://chatbot-b2mcvnqwe9xjcsv43egn9u.streamlit.app  

---

## 🧩 Notes & Tips

- **Similarity direction:** lower distance = more similar.  
  In code, results are currently sorted `reverse=True` (highest distance first).  
  Change to ascending for best-match first:
  ```python
  pairs.sort(key=lambda x: x[1])  # lower = better
  ```

- **Persistence:** Current `Chroma.from_documents` has no `persist_directory`, so index may not survive restart. Use:
  ```python
  vectorstore = Chroma.from_documents(..., persist_directory="chroma_data")
  vectorstore.persist()
  ```

- **Security:** add `.env` and `*.env` to `.gitignore`.

- **Large PDFs:** adjust `chunk_size` and `chunk_overlap` in the text splitter.

---

## 🚀 Deployment

- **Backend**: Render (already live) → https://chatbot-43d0.onrender.com  
- **Frontend**: Streamlit Cloud (already live) → https://chatbot-b2mcvnqwe9xjcsv43egn9u.streamlit.app  

To deploy your own:  
- Backend: Render / Fly.io / Railway / VPS. Ensure `google_api` env var is set.  
- Frontend: Streamlit Cloud. Point `API_BASE` to your deployed API.  

---

## 🧪 Troubleshooting

- **401/403 errors** → check your `google_api` key/quota.  
- **"Only PDF files are allowed."** → upload `.pdf`.  
- **Empty results** → check that PDFs were indexed (use `/upload` or `/reindex`).  
- **Vector count = None** → some backends don’t expose counts.  
- **Free-tier slowness** → large PDFs and cold starts take time.  

---

## 📄 License

Choose your license (MIT / Apache-2.0 / etc).  

---

## 🙌 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://python.langchain.com/)  
- [Chroma](https://www.trychroma.com/)  
- [Google Generative AI](https://ai.google.dev/)  
