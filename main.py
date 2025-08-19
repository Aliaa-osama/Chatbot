from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
from pathlib import Path
import uvicorn

# LangChain / Google GenAI / Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("google_api")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "resumes")

# Default folder for storing PDFs on disk
DEFAULT_CV_DIR = Path(__file__).resolve().parent / "cvs"


app = FastAPI(title="Chatbot + Resume Scorer API (Chroma, PDF via PyPDFLoader)", version="1.0.0")

# ===== MODELS =====

scoring_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)  

# ===== LOADERS =====
load = PyPDFDirectoryLoader(str(DEFAULT_CV_DIR))
doc = load.load()

# ===== TEXT SPLITTER =====
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunk = text_splitter.split_documents(doc)
for doc in chunk:
    src = doc.metadata.get("source", "Unknown CV")
    doc.metadata["file_name"] = Path(src).name

# ===== EMBEDDINGS =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# ===== VECTORSTORE =====
vectorstore = Chroma.from_documents(documents=chunk, embedding=embeddings, collection_name=COLLECTION_NAME)

# ===== PROMPTS =====
score_prompt = ChatPromptTemplate.from_template(
    """You are a resume scoring assistant.
You will be given a job description and one or more CV snippets (may include multiple candidates).
Score how well the CV content matches the job description.
Return:
- A score from 0 to 100 (higher is better)
- A brief explanation focusing on concrete matches and gaps.

Job Description:
{job_description}

CV Content:
{cv}

Return a concise response: "Score: <number>/100
Reason: <one short paragraph>"."""
)

def relevant_docs(query: str, k: int = 2):
    """Retrieve top-k relevant documents with scores and format them."""
    if vectorstore is None:
        return []
    pairs = vectorstore.similarity_search_with_score(query, k=k)
    
    pairs.sort(key=lambda x: x[1], reverse=False)
    formatted = []
    for doc, dist in pairs:
        formatted.append(
            f"CV: {doc.metadata.get('file_name', 'Unknown CV')}\n"
            f"SimilarityDistance: {dist:.4f}\n"
            f"Content: {doc.page_content}"
        )
    return formatted

# === SCORE CV =====
def score_cv(job_description: str, cv_content: str):
    """Use the scoring prompt & LLM to output a score and reason."""
    messages = score_prompt.format_messages(job_description=job_description, cv=cv_content)
    response = scoring_llm.invoke(messages)
    return response.content

# ===== API ENDPOINTS =====
class AskRequest(BaseModel):
    query: str
    k: int = 2



class ReindexRequest(BaseModel):
    directory: Optional[str] = None

@app.post("/ask")
async def ask(request: AskRequest):
    """
    Retrieve relevant CVs and score them.
    """
    if request.query is None or request.query.strip() == "":
        return []

    docs = relevant_docs(request.query, k=request.k)

    results = []
    for d in docs:
        scored = score_cv(request.query, d)
        results.append({
            "cv_snippet": d,
            "score": scored
        })

    return results



# ===== REINDEX =====
def rebuild_vectorstore(pdf_dir: str) -> Chroma:
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    for d in chunks:
        src = d.metadata.get("source", "Unknown CV")
        d.metadata["file_name"] = Path(src).name

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vs

@app.post("/reindex")
async def reindex(req: ReindexRequest):
    """
    Rebuild the vector store from a directory of PDFs.
    Example body: { "directory": "E:/NTI/chatbot/cvs" }
    If 'directory' is omitted, the default path is used.
    """
    target_dir = Path(req.directory or DEFAULT_CV_DIR)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {target_dir}")

    global vectorstore
    vectorstore = rebuild_vectorstore(str(target_dir))

    try:
        count = vectorstore._collection.count()
    except Exception:
        count = None

    return {"status": "reindexed", "directory": str(target_dir), "num_docs": count}

# ======== UPLOAD HELPERS & ENDPOINTS ========

def _unique_path(path: Path) -> Path:
    """Avoid overwriting files: adds _1, _2, ... if name exists."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def _index_pdf_file(pdf_path: Path):
    """Load a single PDF, split to chunks, tag metadata, and add to vectorstore."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    for d in chunks:
        d.metadata["file_name"] = pdf_path.name
    # Add to existing index
    global vectorstore
    if vectorstore is None:
        # Shouldn't happen in your flow, but safe fallback:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=COLLECTION_NAME)
        vectorstore = vs
    else:
        vectorstore.add_documents(chunks)

@app.post("/upload")
async def upload_cv(file: UploadFile = File(...)):
    """
    Upload a single CV PDF and add it to the vectorstore.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    DEFAULT_CV_DIR.mkdir(parents=True, exist_ok=True)

    # Save to disk (avoid overwrite)
    dest = _unique_path(DEFAULT_CV_DIR / Path(file.filename).name)
    data = await file.read()
    dest.write_bytes(data)

    # Index the saved PDF
    try:
        _index_pdf_file(dest)
        count = vectorstore._collection.count()
    except Exception as e:
        # Clean up file on failure (optional)
        # dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return {
        "status": "uploaded",
        "saved_as": str(dest),
        "vector_count": count
    }

@app.post("/upload-batch")
async def upload_cv_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple CV PDFs and add them to the vectorstore.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    DEFAULT_CV_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = _unique_path(DEFAULT_CV_DIR / Path(f.filename).name)
        data = await f.read()
        dest.write_bytes(data)
        try:
            _index_pdf_file(dest)
            saved.append(str(dest))
        except Exception as e:
            # skip problematic file but continue others
            continue

    try:
        count = vectorstore._collection.count()
    except Exception:
        count = None

    if not saved:
        raise HTTPException(status_code=400, detail="No valid PDFs were uploaded or indexed.")
    return {"status": "uploaded", "saved": saved, "vector_count": count}

# ===== HEALTH CHECK =====
@app.get("/health")
async def health():
    return {"status": "ok"}

# ===== DOCS =====
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Chatbot + Resume Scorer API! Use /docs for API documentation."}

# ===== LOCAL RUN =====
if __name__ == "__main__":
   
    uvicorn.run(app, host="127.0.0.1", port=8000)